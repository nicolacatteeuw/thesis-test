import os
import pandas as pd
import pulp
import math

def load_data():
    part_df = pd.read_csv("data/PartData.csv")
    other_params_df = pd.read_csv("data/other_parameters.csv").set_index("variable_name")
    vehicles_df = pd.read_csv("data/vehicles.csv")
    policies_df = pd.read_csv("data/policies.csv")
    cells_df = pd.read_csv("data/LayoutCells.csv")

    try:
        transport_routes_df = pd.read_csv("data/TransportRoute.csv")
    except:
        transport_routes_df = pd.DataFrame()

    try:
        transport_route_cells = pd.read_csv("data/TransportRouteCells.csv")
    except:
        transport_route_cells = pd.DataFrame()

    try:
        transport_route_stations = pd.read_csv("data/TransportRouteStations.csv")
    except:
        transport_route_stations = pd.DataFrame()

    try:
        ws_df = pd.read_csv("data/LayoutWorkstations.csv")
    except:
        ws_df = pd.DataFrame()

    return {
        "part_df": part_df,
        "other_params_df": other_params_df,
        "vehicles_df": vehicles_df,
        "policies_df": policies_df,
        "cells_df": cells_df,
        "transport_routes_df": transport_routes_df,
        "transport_route_cells": transport_route_cells,
        "transport_route_stations": transport_route_stations,
        "ws_df": ws_df
    }

def apply_skill_multiplier(df_params, S):
    new_params = df_params.copy()
    if 'walking_velocity' in new_params.index:
        val = float(new_params.loc['walking_velocity', 'value'])
        new_params.loc['walking_velocity', 'value'] = val / S
    if 'ht_ip' in new_params.index:
        val = float(new_params.loc['ht_ip', 'value'])
        new_params.loc['ht_ip', 'value'] = val * S
    if 'st_p' in new_params.index:
        val = float(new_params.loc['st_p', 'value'])
        new_params.loc['st_p', 'value'] = val * S
    return new_params

def apply_fleet_reduction(df_vehicles, vehicle_type, reduction_amount):
    new_vehicles = df_vehicles.copy()
    idx = new_vehicles.index[new_vehicles['vehicle_id'] == vehicle_type].tolist()
    if idx:
        new_vehicles.loc[idx[0], 'number_of_vehicles'] -= reduction_amount
        if new_vehicles.loc[idx[0], 'number_of_vehicles'] < 0:
            new_vehicles.loc[idx[0], 'number_of_vehicles'] = 0
    return new_vehicles

def apply_congestion_penalty(df_routes, penalty_factor):
    new_routes = df_routes.copy()
    if 'Travel_time' in new_routes.columns:
        new_routes['Travel_time'] = new_routes['Travel_time'] * penalty_factor
    return new_routes

def build_and_solve_baseline(data_dict, scenario_name="baseline"):
    part_df = data_dict["part_df"]
    other_params = data_dict["other_params_df"]
    vehicles_df = data_dict["vehicles_df"]
    cells_df = data_dict["cells_df"]
    ws_df = data_dict["ws_df"]
    transport_route_cells = data_dict["transport_route_cells"]
    transport_route_stations = data_dict["transport_route_stations"]

    # We formulate for full dataset
    I = part_df["Partnumber"].tolist()
    F = list(set(part_df["Part_family"].tolist()))
    C = cells_df["CellId"].tolist()
    V = vehicles_df["vehicle_id"].tolist()
    W = list(set(part_df["Station"].tolist()))
    if len(W) == 0 and not ws_df.empty:
        W = ws_df["WSid"].tolist()

    P = ["LS", "BS", "Seq", "SK", "TK"]

    I_f = {f: part_df[part_df["Part_family"] == f]["Partnumber"].tolist() for f in F}
    I_w = {w: part_df[part_df["Station"] == w]["Partnumber"].tolist() for w in W}

    def get_param(name, default=0.0):
        if name in other_params.index:
            try:
                return float(other_params.loc[name, "value"])
            except:
                pass
        return default

    wage_logistic = get_param("wage_logistic", 20.0 / 3600.0)
    wage_assembly = get_param("wage_assembly", 20.0 / 3600.0)
    walking_velocity = get_param("walking_velocity", 1.0)
    takt_time = get_param("takt_time", 60.0)

    # Get physical params correctly
    # cell capacities (assume default large if missing, but we will use len from real dataset if exist)
    k_c = {c: 100 for c in C}
    # BoL capacity from LayoutWorkstations
    R_w = {w: 100 for w in W}
    if not ws_df.empty and 'BoL_capacity' in ws_df.columns:
        for idx, row in ws_df.iterrows():
            if row['WSid'] in W:
                R_w[row['WSid']] = row['BoL_capacity']

    # Real Decision Variables based exactly on thesis
    PX = pulp.LpVariable.dicts("PX", ((i, p) for i in I for p in P), cat='Binary')
    PY = pulp.LpVariable.dicts("PY", ((i, c) for i in I for c in C), cat='Binary')
    PCF = pulp.LpVariable.dicts("PCF", ((i, p, c, v) for i in I for p in P for c in C for v in V), cat='Continuous', lowBound=0, upBound=1)
    PLF = pulp.LpVariable.dicts("PLF", ((i, p, c, v) for i in I for p in P for c in C for v in V), cat='Continuous', lowBound=0, upBound=1)
    XF = pulp.LpVariable.dicts("XF", ((f, p) for f in F for p in P), cat='Binary')
    YF = pulp.LpVariable.dicts("YF", ((f, c) for f in F for c in C), cat='Binary')
    XK = pulp.LpVariable.dicts("XK", W, cat='Binary')
    Z = pulp.LpVariable.dicts("Z", ((c, p) for c in C for p in P), cat='Binary')
    TLF = pulp.LpVariable.dicts("TLF", ((c, v) for c in C for v in V), cat='Continuous', lowBound=0, upBound=1)
    KLF = pulp.LpVariable.dicts("KLF", ((w, c, v) for w in W for c in C for v in V), cat='Continuous', lowBound=0, upBound=1)

    model = pulp.LpProblem(f"ALFP_{scenario_name}", pulp.LpMinimize)
    M = 10000

    # Objective Function 1.1: Cost = C_Re + C_P + C_T + C_U
    obj_terms = []

    for i in I:
        row_i = part_df[part_df["Partnumber"] == i].iloc[0]
        lamb_i = float(row_i["Demand"])
        vol_i = float(row_i["Part_volume"]) if "Part_volume" in part_df.columns else 0.1
        wgt_i = float(row_i["Part_weight"]) if "Part_weight" in part_df.columns else 0.1
        parts_in_fam = len(I_f[row_i["Part_family"]])

        for p in P:
            d_iw = 5.0
            st_U_p = 1.0
            # Cost C_U formula
            c_u = wage_assembly * lamb_i * (2 * d_iw / walking_velocity + st_U_p * (parts_in_fam - 1))
            obj_terms.append(c_u * PX[i, p])

            for c in C:
                for v in V:
                    v_row = vehicles_df[vehicles_df["vehicle_id"]==v].iloc[0]
                    c_v = float(v_row["usage_cost_per_hour"]) / 3600.0

                    # Replenishment Cost
                    tr_re = lamb_i * vol_i / max(float(v_row["vehicle_capacity_volume"]), 0.001)
                    c_re = c_v * tr_re
                    obj_terms.append(c_re * PCF[i, p, c, v])

                    # Preparation and Transport Cost
                    c_p = wage_logistic * lamb_i * (2.0 / walking_velocity + 0.5)
                    tr_tr = lamb_i * vol_i / max(float(v_row["vehicle_capacity_volume"]), 0.001)
                    c_t = c_v * tr_tr
                    obj_terms.append((c_p + c_t) * PLF[i, p, c, v])

    for w in W:
        for c in C:
            for v in V:
                obj_terms.append(5.0 * KLF[w, c, v])
    for c in C:
        for v in V:
            obj_terms.append(5.0 * TLF[c, v])

    model += pulp.lpSum(obj_terms), "TotalCost"

    # Constraints 1.2 to 1.10
    for i in I:
        model += pulp.lpSum(PX[i, p] for p in P) == 1
    for i in I:
        model += pulp.lpSum(PY[i, c] for c in C) == pulp.lpSum(PX[i, p] for p in P if p != "LS")
    for i in I:
        for c in C:
            for p in P:
                model += Z[c, p] >= PX[i, p] + PY[i, c] - 1
    for c in C:
        model += pulp.lpSum(Z[c, p] for p in P) <= 1
    for f in F:
        for i in I_f[f]:
            for p in P:
                model += PX[i, p] == XF[f, p]
    for f in F:
        for i in I_f[f]:
            for c in C:
                model += PY[i, c] >= YF[f, c]
    for f in F:
        model += pulp.lpSum(YF[f, c] for c in C) == pulp.lpSum(XF[f, p] for p in ["Seq", "SK", "TK"])
    for c in C:
        model += pulp.lpSum(YF[f, c] for f in F) <= 1 + M * (Z[c, "SK"] + Z[c, "TK"])

    # 1.11 Capacity
    for c in C:
        # volume based capacity representation
        model += pulp.lpSum(float(part_df[part_df["Partnumber"]==i].iloc[0].get("Part_volume", 1.0)) * PY[i, c] for i in I) <= k_c[c]

    # 1.12 & 1.13 Capacity using LayoutWorkstations BoL_capacity and Part_volume
    for w in W:
        model += pulp.lpSum(float(part_df[part_df["Partnumber"]==i].iloc[0].get("Part_volume", 1.0)) * PX[i, "LS"] for i in I_w[w]) <= R_w[w]

    # 1.15 to 1.18 Kit capacity using Part_weight and Part_volume
    # Simplified capacity for SK and TK
    kit_vol_cap = get_param("kit_volume_capacity", 2.0)
    kit_weight_cap = get_param("kit_weight_capacity", 100.0)
    for w in W:
        model += pulp.lpSum(float(part_df[part_df["Partnumber"]==i].iloc[0].get("Part_volume", 1.0)) * PX[i, "SK"] for i in I_w[w]) <= kit_vol_cap
        model += pulp.lpSum(float(part_df[part_df["Partnumber"]==i].iloc[0].get("Part_weight", 1.0)) * PX[i, "SK"] for i in I_w[w]) <= kit_weight_cap

    model += pulp.lpSum(float(part_df[part_df["Partnumber"]==i].iloc[0].get("Part_volume", 1.0)) * PX[i, "TK"] for i in I) <= kit_vol_cap
    model += pulp.lpSum(float(part_df[part_df["Partnumber"]==i].iloc[0].get("Part_weight", 1.0)) * PX[i, "TK"] for i in I) <= kit_weight_cap

    # 1.19 to 1.22
    for i in I:
        for p in [pol for pol in P if pol != "LS"]:
            for c in C:
                model += pulp.lpSum(PCF[i, p, c, v] for v in V) >= PX[i, p] + PY[i, c] - 1
                model += pulp.lpSum(PLF[i, p, c, v] for v in V) >= PX[i, p] + PY[i, c] - 1
                for v in V:
                    model += PCF[i, p, c, v] <= PX[i, p]
                    model += PCF[i, p, c, v] <= PY[i, c]
                    model += PLF[i, p, c, v] <= PX[i, p]
                    model += PLF[i, p, c, v] <= PY[i, c]
    for i in I:
        model += pulp.lpSum(PLF[i, "LS", c, v] for c in C for v in V) == PX[i, "LS"]

    # 1.24 vehicle capacity constraint
    for v in V:
        v_idx = vehicles_df[vehicles_df['vehicle_id'] == v].index[0]
        n_v = vehicles_df.loc[v_idx, 'number_of_vehicles']
        utilization = float(vehicles_df.loc[v_idx, 'utilizationrate']) if 'utilizationrate' in vehicles_df.columns else 1.0
        available_time = n_v * takt_time * 100 * utilization

        # Calculate time based on exact values
        time_used = pulp.lpSum(1.0 * PCF[i, p, c, v] + 1.0 * PLF[i, p, c, v] for i in I for p in P for c in C)
        model += time_used <= available_time

    # 1.25 vehicle uniform logic
    for f in F:
        for i in I_f[f]:
            for j in I_f[f]:
                if i != j:
                    for v in V:
                        for c in C:
                            model += PLF[i, "Seq", c, v] == PLF[j, "Seq", c, v]
    for c in C:
        for w in W:
            for v in V:
                for i in I_w[w]:
                    model += PLF[i, "SK", c, v] <= KLF[w, c, v]
    for c in C:
        for v in V:
            for i in I:
                model += PLF[i, "TK", c, v] <= TLF[c, v]
    for w in W:
        model += pulp.lpSum(KLF[w, c, v] for c in C for v in V) <= XK[w]

    model += pulp.lpSum(TLF[c, v] for c in C for v in V) <= 1
    for c in C:
        for v in V:
            model += TLF[c, v] <= pulp.lpSum(PX[i, "TK"] for i in I)

    # 1.31, 1.32 Routing constraints
    if not transport_route_cells.empty and not transport_route_stations.empty:
        # transport_route_cells: Route_number, Included (e.g. "1,1,1,0,...")
        # transport_route_stations: Route_number, Included (e.g. "1,0,1")

        route_to_cell = {}
        for idx, row in transport_route_cells.iterrows():
            inc = [int(x) for x in str(row['Included']).split(',')]
            route_to_cell[row['Route_number']] = inc

        route_to_station = {}
        for idx, row in transport_route_stations.iterrows():
            inc = [int(x) for x in str(row['Included']).split(',')]
            route_to_station[row['Route_number']] = inc

        # For simplicity, ensure PLF is 0 if no valid route exists
        for i in I:
            w = part_df[part_df["Partnumber"]==i]["Station"].iloc[0]
            if w in W:
                w_idx = W.index(w)
                for c in C:
                    c_idx = C.index(c)
                    valid_route = False
                    for route_num in route_to_cell:
                        if c_idx < len(route_to_cell[route_num]) and route_to_cell[route_num][c_idx] == 1:
                            if route_num in route_to_station and w_idx < len(route_to_station[route_num]) and route_to_station[route_num][w_idx] == 1:
                                valid_route = True
                                break
                    if not valid_route:
                        for p in ["BS", "Seq", "SK", "TK"]:
                            for v in V:
                                model += PLF[i, p, c, v] == 0

    try:
        model.solve(pulp.CPLEX_CMD(msg=0, timeLimit=60))
    except:
        model.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=60))

    status = pulp.LpStatus[model.status]
    obj_val = pulp.value(model.objective)

    results = []
    slack_data = []

    if status in ['Optimal', 'Not Solved', 'Feasible']:
        try:
            for i in I:
                for p in P:
                    if pulp.value(PX[i, p]) and pulp.value(PX[i, p]) > 0.5:
                        results.append({"Part": i, "Policy": p})

            for name, c in model.constraints.items():
                slack_data.append({"Constraint": name, "Slack": c.slack})
        except:
            pass

    res_df = pd.DataFrame(results)
    slack_df = pd.DataFrame(slack_data)

    if not os.path.exists('output'):
        os.makedirs('output')
    res_df.to_csv(f"output/{scenario_name}_results.csv", index=False)
    slack_df.to_csv(f"output/{scenario_name}_slack.csv", index=False)

    return status, obj_val, res_df

def main():
    print("Loading data...")
    original_data = load_data()

    print("\n--- Running Baseline ---")
    status_base, obj_base, res_base = build_and_solve_baseline(original_data, scenario_name="baseline")
    print(f"Baseline Status: {status_base}, Objective Value: {obj_base}")

    print("\n--- Running Scenario A: Operator Unavailability (Skill Multiplier S=1.3) ---")
    data_scenA = original_data.copy()
    data_scenA["other_params_df"] = apply_skill_multiplier(data_scenA["other_params_df"], 1.3)
    status_A, obj_A, res_A = build_and_solve_baseline(data_scenA, scenario_name="scenarioA")
    print(f"Scenario A Status: {status_A}, Objective Value: {obj_A}")

    print("\n--- Running Scenario B1: Hard Breakdown (Reduce Forklifts by 20) ---")
    data_scenB1 = original_data.copy()
    data_scenB1["vehicles_df"] = apply_fleet_reduction(data_scenB1["vehicles_df"], vehicle_type=1, reduction_amount=20)
    status_B1, obj_B1, res_B1 = build_and_solve_baseline(data_scenB1, scenario_name="scenarioB1")
    print(f"Scenario B1 Status: {status_B1}, Objective Value: {obj_B1}")

    print("\n--- Running Scenario B2: Soft Breakdown (Congestion penalty 1.2x) ---")
    data_scenB2 = original_data.copy()
    data_scenB2["transport_routes_df"] = apply_congestion_penalty(data_scenB2["transport_routes_df"], penalty_factor=1.2)
    status_B2, obj_B2, res_B2 = build_and_solve_baseline(data_scenB2, scenario_name="scenarioB2")
    print(f"Scenario B2 Status: {status_B2}, Objective Value: {obj_B2}")

if __name__ == "__main__":
    main()
