import pandas as pd
import numpy as np

from topomap import TopoMap
from validate import validate_points

data = pd.read_csv("data\iris.csv")

x = data.iloc[:,:3]
y = data.iloc[:,-1]

topo = TopoMap()

out = topo.transform(x)

print("\nTransformed points:\n", out.head())

print(f"\nTotal time taken to run transformation: {topo.total_time:4.4f} secs")

print("\nPloting the results and saving to path 'img.png'")

topo.plot(y, 'img.png')
print('\nValidating the Rips guarantee for H0 values')

x = np.array(x)
tx = topo._tpoints
print(validate_points(x, tx, verbose = 0)) #True if H0 values match
