import pandas as pd
import numpy as np
import pulp
import os
import math

def load_data(data_dir):
    dfs = {}
    expected_files = [
        "LayoutCells.csv",
        "LayoutLimits.csv",
        "LayoutWarehouse.csv",
        "LayoutWorkstations.csv",
        "LinestockingRoute.csv",
        "LinestockingRouteStations.csv",
        "other parameters.csv",
        "PartData.csv",
        "policies.csv",
        "ReplenishmentRoute.csv",
        "ReplenishmentRouteCells.csv",
        "strategic_layout_output.csv",
        "strategic_overview.csv",
        "TransportRoute.csv",
        "TransportRouteCells.csv",
        "TransportRouteStations.csv",
        "vehicles strategic.csv",
        "vehicles.csv",
        # Including legacy dummy names temporarily to prevent breaking any existing logic
        "part_policy_data.csv", "cells.csv", "stations.csv", "cell_station_distances.csv",
        "parameters.csv", "transport_times.csv", "kit_times.csv", "tk_times.csv"
    ]
    loaded_count = 0
    for file in expected_files:
        filepath = os.path.join(data_dir, file)
        if os.path.exists(filepath):
            key = file.replace(".csv", "")
            try:
                dfs[key] = pd.read_csv(filepath)
                loaded_count += 1
            except Exception as e:
                print(f"Error loading {file}: {e}")

    print(f"Successfully loaded {loaded_count} out of {len(expected_files)} expected CSV files from the '{data_dir}' directory.")
    if loaded_count == 0:
        print(f"WARNING: No CSV files were found in the '{os.path.abspath(data_dir)}' directory. Please ensure your CSV files are placed inside a folder named '{data_dir}'.")

    # To keep backward compatibility with existing model mapping in the script:
    if "PartData" in dfs: dfs["part_data"] = dfs["PartData"]
    if "TransportRoute" in dfs: dfs["transport_route"] = dfs["TransportRoute"]

    return dfs

def build_and_solve_baseline(data_dict, scenario_name="baseline", output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Sets Extraction
    I = []
    if "part_data" in data_dict:
        part_col = 'i' if 'i' in data_dict["part_data"].columns else 'Partnumber'
        if part_col in data_dict["part_data"].columns:
            I = data_dict["part_data"][part_col].unique().tolist()

    P = []
    if "policies" in data_dict:
        p_col = 'p' if 'p' in data_dict["policies"].columns else 'Policy'
        if p_col in data_dict["policies"].columns:
            P = data_dict["policies"][p_col].unique().tolist()
    C = []
    if "LayoutCells" in data_dict:
        c_col = 'c' if 'c' in data_dict["LayoutCells"].columns else 'Cell'
        if c_col in data_dict["LayoutCells"].columns:
            C = data_dict["LayoutCells"][c_col].unique().tolist()

    W = []
    if "LayoutWorkstations" in data_dict:
        w_col = 'w' if 'w' in data_dict["LayoutWorkstations"].columns else 'Station'
        if w_col in data_dict["LayoutWorkstations"].columns:
            W = data_dict["LayoutWorkstations"][w_col].unique().tolist()

    V = []
    if "vehicles" in data_dict:
        v_col = 'vehicle_id' if 'vehicle_id' in data_dict["vehicles"].columns else 'Vehicle'
        if v_col in data_dict["vehicles"].columns:
            V = data_dict["vehicles"][v_col].unique().tolist()

    F_all = []
    if "part_data" in data_dict:
        f_col = 'f' if 'f' in data_dict["part_data"].columns else 'Part_family'
        if f_col in data_dict["part_data"].columns:
            F_all = data_dict["part_data"][f_col].unique().tolist()

    F = [f for f in F_all if str(f) != 'None' and str(f) != 'nan'] # exclude 'None' and nan family

    # Map parts to families and stations
    I_f = {}
    I_w = {}
    F_w = {}

    if "part_data" in data_dict:
        part_col = 'i' if 'i' in data_dict["part_data"].columns else 'Partnumber'
        f_col = 'f' if 'f' in data_dict["part_data"].columns else 'Part_family'
        w_col = 'w' if 'w' in data_dict["part_data"].columns else 'Station'

        for f in F:
            I_f[f] = data_dict["part_data"][data_dict["part_data"][f_col] == f][part_col].tolist()

        for w in W:
            if w_col in data_dict["part_data"].columns:
                I_w[w] = data_dict["part_data"][data_dict["part_data"][w_col] == w][part_col].tolist()
            else:
                I_w[w] = []

        for w in W:
            if w_col in data_dict["part_data"].columns:
                f_in_w = data_dict["part_data"][data_dict["part_data"][w_col] == w][f_col].tolist()
                F_w[w] = [f for f in list(set(f_in_w) & set(F)) if len(I_f.get(f, [])) > 0]
            else:
                F_w[w] = []

    # Parameters
    # Map number of available vehicles n_v
    n_v = {}
    if "vehicles" in data_dict:
        v_col = 'vehicle_id' if 'vehicle_id' in data_dict["vehicles"].columns else 'Vehicle'
        num_col = 'number_of_vehicles' if 'number_of_vehicles' in data_dict["vehicles"].columns else 'Number_of_vehicles'
        if v_col in data_dict["vehicles"].columns and num_col in data_dict["vehicles"].columns:
            n_v = dict(zip(data_dict["vehicles"][v_col], data_dict["vehicles"][num_col]))
    T = 480
    M_big = 10000 # Big M
    # In user files, "other parameters" seems to be the parameters file.
    if "other parameters" in data_dict:
        param_df = data_dict["other parameters"]
        p_col = 'param' if 'param' in param_df.columns else 'Parameter'
        v_col = 'value' if 'value' in param_df.columns else 'Value'
        if p_col in param_df.columns and v_col in param_df.columns and "T" in param_df[p_col].values:
            T = float(param_df[param_df[p_col] == 'T'][v_col].values[0])
    elif "parameters" in data_dict:
        param_df = data_dict["parameters"]
        if "param" in param_df.columns and "T" in param_df['param'].values:
            T = float(param_df[param_df['param'] == 'T']['value'].values[0])

    # Transport times logic, fallback to new user files.
    # The actual transport times likely have to be mapped from ReplenishmentRoute or TransportRoute,
    # but since we lack the exact definition of 't_Re', 't_Tr', 't_SK', 't_TK' in those CSVs, we use
    # dictionary fallback logic.
    t_Re = {}
    t_Tr = {}
    if "transport_times" in data_dict:
        for idx, row in data_dict["transport_times"].iterrows():
            t_Re[(row['i'], row['p'], row['c'], row['v'])] = row['t_Re']
            t_Tr[(row['i'], row['p'], row['c'], row['v'])] = row['t_Tr']

    t_SK = {}
    if "kit_times" in data_dict:
        for idx, row in data_dict["kit_times"].iterrows():
            t_SK[(row['w'], row['c'], row['v'])] = row['t_SK']

    t_TK = {}
    if "tk_times" in data_dict:
        for idx, row in data_dict["tk_times"].iterrows():
            t_TK[(row['c'], row['v'])] = row['t_TK']

    # BoL capacity and Space parameters
    k_C_c = {}
    if "LayoutCells" in data_dict:
        c_col = 'c' if 'c' in data_dict["LayoutCells"].columns else 'Cell'
        cap_col = 'L_c_cell' if 'L_c_cell' in data_dict["LayoutCells"].columns else 'Capacity'
        if c_col in data_dict["LayoutCells"].columns and cap_col in data_dict["LayoutCells"].columns:
            k_C_c = dict(zip(data_dict["LayoutCells"][c_col], data_dict["LayoutCells"][cap_col]))

    k_W_w = {}
    if "LayoutWorkstations" in data_dict:
        w_col = 'w' if 'w' in data_dict["LayoutWorkstations"].columns else 'Station'
        cap_col = 'L_w_BoL' if 'L_w_BoL' in data_dict["LayoutWorkstations"].columns else 'Capacity'
        if w_col in data_dict["LayoutWorkstations"].columns and cap_col in data_dict["LayoutWorkstations"].columns:
            k_W_w = dict(zip(data_dict["LayoutWorkstations"][w_col], data_dict["LayoutWorkstations"][cap_col]))

    # Dummy parameters for capacity constraints (fallback to 1 if not provided)
    k_P_L = {i: 1 for i in I}
    k_P_B = 1 # space per boxed supply rack
    k_P_S = 1 # space per sequencing
    k_P_K = 1 # space per kit
    bs_rack_size = 10 # parts per BS rack

    # Route extraction
    TransportationRoute = {}
    if "transport_route" in data_dict:
        for idx, row in data_dict["transport_route"].iterrows():
            TransportationRoute[(row['r'], row['c'], row['w'])] = row['Valid']

    # Kit capacity parameters
    k_FV = {f: 1 for f in F} # volume of family f
    k_FW = {f: 1 for f in F} # weight of family f
    k_KV = 10 # Kit volume capacity
    k_KW = 10 # Kit weight capacity

    # 2. Model Initialization
    model = pulp.LpProblem(f"ALFP_Model_{scenario_name}", pulp.LpMinimize)

    # Decision Variables
    PX = pulp.LpVariable.dicts("PX", ((i, p) for i in I for p in P), cat="Binary")
    PY = pulp.LpVariable.dicts("PY", ((i, c) for i in I for c in C), cat="Binary")
    PCF = pulp.LpVariable.dicts("PCF", ((i, p, c, v) for i in I for p in P for c in C for v in V), cat="Continuous", lowBound=0)
    PLF = pulp.LpVariable.dicts("PLF", ((i, p, c, v) for i in I for p in P for c in C for v in V), cat="Continuous", lowBound=0)
    KLF = pulp.LpVariable.dicts("KLF", ((w, c, v) for w in W for c in C for v in V), cat="Continuous", lowBound=0)
    TLF = pulp.LpVariable.dicts("TLF", ((c, v) for c in C for v in V), cat="Continuous", lowBound=0)

    Z = pulp.LpVariable.dicts("Z", ((c, p) for c in C for p in P), cat="Binary")
    X_F = pulp.LpVariable.dicts("X_F", ((f, p) for f in F for p in P), cat="Binary")
    Y_F = pulp.LpVariable.dicts("Y_F", ((f, c) for f in F for c in C), cat="Binary")

    X_K = pulp.LpVariable.dicts("X_K", (w for w in W), cat="Integer", lowBound=0)
    bsracks = pulp.LpVariable.dicts("bsracks", (w for w in W), cat="Integer", lowBound=0)

    # Objective Function
    model += pulp.lpSum(PCF[i, p, c, v] for i in I for p in P for c in C for v in V) + \
             pulp.lpSum(PLF[i, p, c, v] for i in I for p in P for c in C for v in V) + \
             pulp.lpSum(KLF[w, c, v] for w in W for c in C for v in V) + \
             pulp.lpSum(TLF[c, v] for c in C for v in V) + \
             pulp.lpSum(PX[i, p] for i in I for p in P)

    # Constraint 1.2
    for i in I:
        model += pulp.lpSum(PX[i, p] for p in P) == 1

    # Constraint 1.3
    for i in I:
        p_not_LS = [p for p in P if p != 'LS']
        model += pulp.lpSum(PY[i, c] for c in C) == pulp.lpSum(PX[i, p] for p in p_not_LS)

    # Constraint 1.4, 1.5, 1.6
    for c in C:
        for p in P:
            for i in I:
                model += Z[c, p] >= PX[i, p] + PY[i, c] - 1
        model += pulp.lpSum(Z[c, p] for p in P) <= 1
        model += pulp.lpSum(Z[c, p] for p in P) <= pulp.lpSum(PY[i, c] for i in I)

    # Constraint 1.7, 1.8
    for f in F:
        for i in I_f[f]:
            for p in P:
                model += PX[i, p] == X_F[f, p]
            for c in C:
                model += PY[i, c] >= Y_F[f, c]

    # Constraint 1.9
    for f in F:
        p_not_LS_BS = [p for p in P if p not in ['LS', 'BS']]
        model += pulp.lpSum(Y_F[f, c] for c in C) == pulp.lpSum(X_F[f, p] for p in p_not_LS_BS)

    # Constraint 1.10
    for c in C:
        Z_cSK = Z[c, 'SK'] if 'SK' in P else 0
        Z_cTK = Z[c, 'TK'] if 'TK' in P else 0
        model += pulp.lpSum(Y_F[f, c] for f in F) <= 1 + M_big * (Z_cSK + Z_cTK)

    # Constraint 1.11
    for c in C:
        model += pulp.lpSum(k_P_L[i] * PY[i, c] for i in I) <= k_C_c.get(c, 1000)

    # Constraint 1.12 & 1.13 & Linearization
    for w in W:
        term_L = pulp.lpSum(X_F[f, 'LS'] * k_P_L[I_f[f][0]] * len(I_f[f]) for f in F_w[w] if len(I_f[f]) > 0) if 'LS' in P else 0
        term_S = pulp.lpSum(X_F[f, 'Seq'] * k_P_S for f in F_w[w]) if 'Seq' in P else 0

        model += term_L + bsracks[w] * k_P_B + term_S + X_K[w] * k_P_K <= k_W_w.get(w, 1000)

        if 'BS' in P:
            sum_BS = pulp.lpSum(X_F[f, 'BS'] * len(I_f[f]) / bs_rack_size for f in F_w[w])
            model += bsracks[w] >= sum_BS
            model += bsracks[w] - 1 <= sum_BS

    # Constraint 1.14
    for w in W:
        model += X_K[w] == pulp.lpSum(KLF[w, c, v] for c in C for v in V)

    # Constraint 1.15 & 1.16
    if 'TK' in P:
        model += pulp.lpSum(k_FV[f] * X_F[f, 'TK'] for f in F) <= k_KV
        model += pulp.lpSum(k_FW[f] * X_F[f, 'TK'] for f in F) <= k_KW

    # Constraint 1.17 & 1.18
    for w in W:
        for c in C:
            Z_cSK = Z[c, 'SK'] if 'SK' in P else 0
            model += pulp.lpSum(k_FV[f] * Y_F[f, c] for f in F_w[w]) <= k_KV + M_big * (1 - Z_cSK)
            model += pulp.lpSum(k_FW[f] * Y_F[f, c] for f in F_w[w]) <= k_KW + M_big * (1 - Z_cSK)

    # Constraints 1.19 - 1.22
    for i in I:
        for p in [pol for pol in P if pol != 'LS']:
            for c in C:
                model += pulp.lpSum(PCF[i, p, c, v] for v in V) >= PX[i, p] + PY[i, c] - 1
                model += pulp.lpSum(PLF[i, p, c, v] for v in V) >= PX[i, p] + PY[i, c] - 1
                model += pulp.lpSum(PCF[i, p, c, v] for v in V) <= (PX[i, p] + PY[i, c]) / 2
                model += pulp.lpSum(PLF[i, p, c, v] for v in V) <= (PX[i, p] + PY[i, c]) / 2

    # Constraint 1.23
    if 'LS' in P:
        for i in I:
            model += pulp.lpSum(PLF[i, 'LS', c, v] for c in C for v in V) == PX[i, 'LS']

    # Constraint 1.24
    for v in V:
        term1 = pulp.lpSum(PCF[i, p, c, v] * t_Re.get((i, p, c, v), 0) for i in I for p in P for c in C)
        term2 = pulp.lpSum(PLF[i, p, c, v] * t_Tr.get((i, p, c, v), 0) for i in I for p in ['LS', 'BS', 'Seq'] if p in P for c in C)
        term3 = pulp.lpSum(KLF[w, c, v] * t_SK.get((w, c, v), 0) for w in W for c in C)
        term4 = pulp.lpSum(TLF[c, v] * t_TK.get((c, v), 0) for c in C)

        model += term1 + term2 + term3 + term4 <= n_v.get(v, 0) * T

    # Constraint 1.25
    for f in F:
        if 'Seq' in P:
            for i in I_f[f]:
                for j in I_f[f]:
                    if i != j:
                        for c in C:
                            for v in V:
                                model += PLF[i, 'Seq', c, v] == PLF[j, 'Seq', c, v]

    # Constraint 1.26
    if 'SK' in P:
        for w in W:
            for i in I_w[w]:
                for c in C:
                    for v in V:
                        model += PLF[i, 'SK', c, v] <= KLF[w, c, v]

    # Constraint 1.27
    if 'TK' in P:
        for i in I:
            for c in C:
                for v in V:
                    model += PLF[i, 'TK', c, v] <= TLF[c, v]

    # Constraint 1.28
    for w in W:
        model += pulp.lpSum(KLF[w, c, v] for c in C for v in V) <= X_K[w]

    # Constraint 1.29
    # Limit number of traveling kits in the system if necessary, here assumed \sum_{cv} TLF_cv <= limit. Using 1 as per formula.
    model += pulp.lpSum(TLF[c, v] for c in C for v in V) <= 1

    # Constraint 1.30
    if 'TK' in P:
        model += pulp.lpSum(TLF[c, v] for c in C for v in V) <= pulp.lpSum(PX[i, 'TK'] for i in I)

    # Constraints 1.31
    for v in V:
        for w in W:
            for i in I_w.get(w, []):
                for p in ['BS', 'Seq', 'SK']:
                    if p in P:
                        for c in C:
                            model += PLF[i, p, c, v] <= pulp.lpSum(TransportationRoute.get((r, c, w), 0) for r in set([route[0] for route in TransportationRoute.keys()]))

    # Constraint 1.32
    if 'TK' in P:
        for v in V:
            for i in I:
                for c in C:
                    model += PLF[i, 'TK', c, v] <= pulp.lpSum(TransportationRoute.get((r, c, w), 0) for w in W for r in set([route[0] for route in TransportationRoute.keys()]))

    # Solve
    solver = pulp.CPLEX_CMD(msg=False)
    try:
        model.solve(solver)
    except:
        model.solve(pulp.PULP_CBC_CMD(msg=False))

    obj_val = pulp.value(model.objective)
    status_str = pulp.LpStatus[model.status]

    # Save results
    results = []
    if obj_val is not None:
        for i in I:
            for p in P:
                if pulp.value(PX[i, p]) and pulp.value(PX[i, p]) > 0.5:
                    results.append({'i': i, 'Variable': 'PX', 'Value': p})
        for i in I:
            for c in C:
                if pulp.value(PY[i, c]) and pulp.value(PY[i, c]) > 0.5:
                    results.append({'i': i, 'Variable': 'PY', 'Value': c})

    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(output_dir, f"{scenario_name}_results.csv"), index=False)

    return model, res_df, obj_val, status_str

def apply_skill_multiplier(data_dict, S, target_cell):
    new_dict = {k: v.copy() for k, v in data_dict.items()}

    param_key = "other parameters" if "other parameters" in new_dict else "parameters"
    if param_key in new_dict:
        params = new_dict[param_key]
        p_col = 'param' if 'param' in params.columns else 'Parameter'
        v_col = 'value' if 'value' in params.columns else 'Value'

        if p_col in params.columns and 'OV' in params[p_col].values:
            ov_idx = params.index[params[p_col] == 'OV'][0]
            params.at[ov_idx, v_col] = float(params.at[ov_idx, v_col]) / S

    if "policies" in new_dict:
        pols = new_dict["policies"]
        if 'st_p' in pols.columns:
            pols['st_p'] = pols['st_p'] * S

    if "part_policy_data" in new_dict:
        ppd = new_dict["part_policy_data"]
        if 'ht_ip' in ppd.columns:
            ppd['ht_ip'] = ppd['ht_ip'] * S

    if "transport_times" in new_dict:
        tt = new_dict["transport_times"]
        mask = tt['c'] == target_cell
        tt.loc[mask, 't_Re'] = tt.loc[mask, 't_Re'] * S

    return new_dict

def apply_fleet_reduction(data_dict, vehicle_type, reduction_amount):
    new_dict = {k: v.copy() for k, v in data_dict.items()}
    if "vehicles" in new_dict:
        vehs = new_dict["vehicles"]
        v_col = 'vehicle_id' if 'vehicle_id' in vehs.columns else 'Vehicle'
        num_col = 'number_of_vehicles' if 'number_of_vehicles' in vehs.columns else 'Number_of_vehicles'

        if v_col in vehs.columns and num_col in vehs.columns:
            mask = vehs[v_col] == vehicle_type
            vehs.loc[mask, num_col] = np.maximum(0, vehs.loc[mask, num_col] - reduction_amount)
    return new_dict

def apply_congestion_penalty(data_dict, penalty_factor):
    new_dict = {k: v.copy() for k, v in data_dict.items()}
    if "transport_times" in new_dict:
        new_dict["transport_times"]['t_Tr'] *= penalty_factor
    if "kit_times" in new_dict:
        new_dict["kit_times"]['t_SK'] *= penalty_factor
    if "tk_times" in new_dict:
        new_dict["tk_times"]['t_TK'] *= penalty_factor
    return new_dict

def main():
    print("Loading data...")
    data_dict = load_data('data')

    print("Running Baseline Model...")
    baseline_model, baseline_res, baseline_obj, baseline_status = build_and_solve_baseline(data_dict, scenario_name="baseline")
    print(f"Baseline Status: {baseline_status} | Objective: {baseline_obj}")

    print("Running Scenario A: Operator Unavailability...")
    data_scen_a = apply_skill_multiplier(data_dict, S=1.3, target_cell='C1')
    model_a, res_a, obj_a, status_a = build_and_solve_baseline(data_scen_a, scenario_name="scenario_A")
    print(f"Scenario A Status: {status_a} | Objective: {obj_a}")

    print("Running Scenario B1: Fleet Reduction...")
    data_scen_b1 = apply_fleet_reduction(data_dict, vehicle_type='V1', reduction_amount=1)
    model_b1, res_b1, obj_b1, status_b1 = build_and_solve_baseline(data_scen_b1, scenario_name="scenario_B1")
    print(f"Scenario B1 Status: {status_b1} | Objective: {obj_b1}")

    print("Running Scenario B2: Congestion Penalty...")
    data_scen_b2 = apply_congestion_penalty(data_dict, penalty_factor=1.2)
    model_b2, res_b2, obj_b2, status_b2 = build_and_solve_baseline(data_scen_b2, scenario_name="scenario_B2")
    print(f"Scenario B2 Status: {status_b2} | Objective: {obj_b2}")

if __name__ == "__main__":
    main()
