import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns; sns.set()
import csv

df = pd.read_csv('Dog combined.csv')
features = df[['Latitude', 'Longitude']]
X = np.array(features)
print(X[:10])

from k_means_constrained import KMeansConstrained
sse = []
for k in range(8, 25):
  clf = KMeansConstrained(
  n_clusters=k,
  size_min=6,
  random_state=0
  )
  clf.fit(X)
  sse.append(clf.inertia_)

plt.style.use("fivethirtyeight")
plt.plot(range(8, 25), sse)
plt.xticks(range(8, 25))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()

from k_means_constrained import KMeansConstrained
silhouette_coefficients = []
for k in range(8, 25):
  clf = KMeansConstrained(
  n_clusters=k,
  size_min=6,
  random_state=0
  )
  clf.fit(X)
  score = silhouette_score(X, clf.labels_)
  silhouette_coefficients.append(score)

plt.style.use("fivethirtyeight")
plt.plot(range(8, 25), silhouette_coefficients)
plt.xticks(range(8, 25))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
plt.show()

clf = KMeansConstrained(
  n_clusters=16,
  size_min=6,
  random_state=0
  )
clf.fit_predict(X)
# save results


labels = clf.labels_
centers = clf.cluster_centers_ # Coordinates of cluster centers.
# send back into dataframe and display it
df['cluster'] = labels
# display the number of mamber each clustering
_clusters = df.groupby('cluster').count()
print(_clusters)

df.plot.scatter(x = 'Latitude', y = 'Longitude', c=labels, s=5,cmap='viridis' )
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=800, alpha=0.5)
print(centers)

pd.DataFrame(centers).to_csv('samplecomb.csv')
clusterCount = np.bincount(labels)
pd.DataFrame(clusterCount).to_csv('countcomb.csv')
