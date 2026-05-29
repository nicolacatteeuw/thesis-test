import pandas as pd
print("TransportRouteCells")
try: print(pd.read_csv("data/TransportRouteCells.csv").head(2))
except: print("missing")

print("TransportRouteStations")
try: print(pd.read_csv("data/TransportRouteStations.csv").head(2))
except: print("missing")
