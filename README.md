# Assembly Line Feeding Problem (ALFP) - MILP Model and Disruption DoE

This project models an Assembly Line Feeding Problem using Mixed Integer Linear Programming (MILP) to evaluate the impact of short-term disruptions (such as operator unavailability and vehicle breakdown).

## Structure
- `alfp_model.py`: Main executable file containing:
  - Data ingestion function
  - MILP model construction function (Deterministic Baseline Model)
  - Data mutation functions (Disruption Scenario Generator - DoE)
  - Execution loop that runs the baseline, generates scenarios, and compares results
- `data/`: Directory containing all required CSV files extracted from the data inputs.

## Setup
```bash
pip install pandas pulp
python alfp_model.py
```

## Model
The model runs with CPLEX command `pulp.CPLEX_CMD()` with a fallback to the default CBC solver if CPLEX is unavailable in the environment. Results are written into the `data/` directory.
