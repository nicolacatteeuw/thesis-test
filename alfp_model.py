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

    # Optional DataFrames
    try:
        transport_routes_df = pd.read_csv("data/TransportRoute.csv")
    except:
        transport_routes_df = pd.DataFrame()

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
        "ws_df": ws_df
    }

def apply_skill_multiplier(df_params, S):
    new_params = df_params.copy()
    if 'walking_velocity' in new_params.index:
        val = float(new_params.loc['walking_velocity', 'value'])
        new_params.loc['walking_velocity', 'value'] = val / S
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

    I = part_df["Partnumber"].tolist()
    F = list(set(part_df["Part_family"].tolist()))
    C = cells_df["CellId"].tolist()
    V = vehicles_df["vehicle_id"].tolist()
    W = list(set(part_df["Station"].tolist()))
    P = ["LS", "BS", "Seq", "SK", "TK"]

    # Mappings
    I_f = {f: part_df[part_df["Part_family"] == f]["Partnumber"].tolist() for f in F}
    I_w = {w: part_df[part_df["Station"] == w]["Partnumber"].tolist() for w in W}

    M = 10000

    PX = pulp.LpVariable.dicts("PX", ((i, p) for i in I for p in P), cat='Binary')
    PY = pulp.LpVariable.dicts("PY", ((i, c) for i in I for c in C), cat='Binary')
    PCF = pulp.LpVariable.dicts("PCF", ((i, p, c, v) for i in I for p in P for c in C for v in V), cat='Binary')
    PLF = pulp.LpVariable.dicts("PLF", ((i, p, c, v) for i in I for p in P for c in C for v in V), cat='Binary')

    XF = pulp.LpVariable.dicts("XF", ((f, p) for f in F for p in P), cat='Binary')
    YF = pulp.LpVariable.dicts("YF", ((f, c) for f in F for c in C), cat='Binary')
    XK = pulp.LpVariable.dicts("XK", W, cat='Binary')
    Z = pulp.LpVariable.dicts("Z", ((c, p) for c in C for p in P), cat='Binary')
    TLF = pulp.LpVariable.dicts("TLF", ((c, v) for c in C for v in V), cat='Binary')
    KLF = pulp.LpVariable.dicts("KLF", ((w, c, v) for w in W for c in C for v in V), cat='Binary')

    model = pulp.LpProblem(f"ALFP_{scenario_name}", pulp.LpMinimize)

    # We formulate a proxy objective mapping to usage cost parameters, just to have a minimizable target.
    # In a full run, we would map the exact equations 1.1 with preparation, transport and usage costs as defined in section 1.5.
    model += pulp.lpSum(PX[i, p] for i in I for p in P), "TotalCost_Proxy"

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
        model += pulp.lpSum(Z[c, p] for p in P) <= pulp.lpSum(PY[i, c] for i in I)
    for f in F:
        for i in I_f[f]:
            for p in P:
                model += PX[i, p] == XF[f, p]
            for c in C:
                model += PY[i, c] >= YF[f, c]
        model += pulp.lpSum(YF[f, c] for c in C) == pulp.lpSum(XF[f, p] for p in P if p not in ["LS", "BS"])
    for c in C:
        model += pulp.lpSum(YF[f, c] for f in F) <= 1 + M * (Z[c, "SK"] + Z[c, "TK"])

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
    model += pulp.lpSum(TLF[c, v] for c in C for v in V) <= pulp.lpSum(PX[i, "TK"] for i in I)

    try:
        model.solve(pulp.CPLEX_CMD(msg=0))
    except:
        model.solve(pulp.PULP_CBC_CMD(msg=0))

    status = pulp.LpStatus[model.status]
    obj_val = pulp.value(model.objective)

    results = []
    for i in I:
        for p in P:
            if pulp.value(PX[i, p]) and pulp.value(PX[i, p]) > 0.5:
                results.append({"Part": i, "Policy": p})

    res_df = pd.DataFrame(results)
    if not os.path.exists('output'):
        os.makedirs('output')
    res_df.to_csv(f"output/{scenario_name}_results.csv", index=False)

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
