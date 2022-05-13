import pandas as pd
import numpy as np

from topomap import TopoMap
from validate import validate_points

output_arr= []
non_par_arr = []

input_path = "data/"
dataset = ["seeds.csv", "mFeat.csv"]
# dataset = ["seeds.csv"]


dataset_final = []
DISTANCE_TYPES = ['euclidean', 'manhattan',  'minkowski']
# DISTANCE_TYPES = ['euclidean']



# x = data.iloc[:,:3]
# y = data.iloc[:,-1]

# topo = TopoMap()

# out = topo.transform(x)

# print("\nTransformed points:\n", out.head())

# print(f"\nTotal time taken to run transformation: {topo.total_time:4.4f} secs")

# print("\nPloting the results and saving to path 'img.png'")

# topo.plot(y, 'img.png')
# print('\nValidating the Rips guarantee for H0 values')

# x = np.array(x)
# tx = topo._tpoints
# print(validate_points(x, tx, verbose = 0)) #True if H0 values match

for data in dataset:
  dataset_final.append(input_path+data)

imagenames = []
i = 0
for filename in dataset:
  path = input_path + filename

  data = pd.read_csv(path)
  data = data.drop_duplicates(keep='first')
  x = data.iloc[:,2:6]
  y = data.iloc[:,-1]
  i = i + 1
  for dtype in DISTANCE_TYPES:
    topo1 = TopoMap(use_parallel=False)
    topo2 = TopoMap(use_parallel=True)


    out1 = topo1.transform(x, distance_type=dtype)
    out2 = topo2.transform(x, distance_type=dtype)

    time1 = round(topo1.total_time,4)
    time2 = round(topo2.total_time,4)
    

    filname = filename.rsplit( ".", 1)[0]
    output_arr.append([filename, time1, time2])
    non_par_arr.append([filename, time1, dtype])


    

    path = 'output_images/img_' + str(filename) + '_' + str(dtype) + '.png'
    
    topo1.plot(y,path)
    imagenames.append(path)



    



output1 = pd.DataFrame(np.array(output_arr),columns=['Dataset','Non Parallel Time','Parallel Time'])
print("*"*60)
output2 = pd.DataFrame(np.array(non_par_arr),columns=['Dataset','Time','Distance'])
print(output1)
print("*"*60)

print(output2)