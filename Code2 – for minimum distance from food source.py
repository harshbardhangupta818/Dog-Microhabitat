import haversine as hs
import pandas as pd
import numpy as np
from haversine import Unit

df = pd.read_csv('Dog day3.csv')
features = df[['Latitude', 'Longitude']]
X = np.array(features)
print(X[:10])

df2 = pd.read_csv('Food_Source.csv')
features = df2[['Latitude', 'Longitude']]
Y = np.array(features)

dis_mat = [ [0]* len(Y) for i in range(len(X))]
for i in range(len(X)):
  for j in range(len(Y)):
    dis= hs.haversine(X[i],Y[j],unit=Unit.METERS)
    dis_mat[i][j]=dis
  print(dis_mat[:10])
  
dis_list= []
for i in range(len(X)):
  dis_list.append(min(dis_mat[i]))
dis_list

pd.DataFrame(dis_list).to_csv('dis_meas_day3.csv')
