import csv
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

rows=pd.read_csv('stars.csv')

# with open('stars.csv','r') as f:
#     df=csv.reader(f)
#     for i in df:
#         rows.append(i)

# headers=rows[0]
# stardata=rows[1:]

# star_masses=[]
# star_radius=[]

# for i in stardata:
#     star_masses.append(i[3])
#     star_radius.append(i[4])

X = rows.iloc[:,[3,4]].values
# for index, star_mass in enumerate(star_masses):
#   temp_list = [star_radius[index], star_mass]
#   X.append(temp_list)

WCSS = []
for i in range(1,11):
  k_means = KMeans(n_clusters=i, init='k-means++', random_state=42)
  k_means.fit(X)
  WCSS.append((k_means.inertia_))

plt.figure(figsize=(10,5))
sns.lineplot(range(1,11), WCSS, marker='o', color='red')
plt.title("Elbow method")
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()