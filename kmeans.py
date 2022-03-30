# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 00:21:41 2021

@author: REKHA
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


data=pd.read_csv("kmeans.csv")
choice=data.head()
print(choice)
choice1=data.shape
print(choice1)
choice3=data.describe()
print(choice3)
choice4=data.dtypes
print(choice4)

choice5=data.isnull().sum()
print(choice5)

# Age Vs Spending Score

plt.scatter(data['Age'],data['Spending Score (1-100)'])
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.grid(linestyle='--')
plt.show()


x=data[['Age', 'Spending Score (1-100)']]
wcss =[]
for i in range(1,10):
    kmeans=KMeans(n_clusters = i, init = 'k-means++', max_iter=300, n_init=10,random_state= 0 )
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    

no_of_clusters=range(1,10)
plt.plot(no_of_clusters,wcss)
plt.grid(linestyle='--')
plt.show()

kmeans=KMeans(4)
identified_clusters=kmeans.fit_predict(x)
identified_clusters

table_with_clustes=data.copy()
table_with_clustes['Clusters']=identified_clusters
table_with_clustes

plt.scatter(table_with_clustes['Age'],table_with_clustes['Spending Score (1-100)'],c=table_with_clustes['Clusters'],cmap='rainbow')
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.title('Plot with Clusters')
plt.grid(linestyle='--')
plt.show()

from sklearn.metrics import silhouette_score
print(f'Silhouette Score(n=4): {silhouette_score(x,identified_clusters)}')


plt.figure(1,figsize=(10,5))
ax=sns.countplot(x='Gender',data=data)
for p in ax.patches:
    ax.annotate('{:}'.format(p.get_height()),(p.get_x()+0.3,p.get_height()))
plt.show()


  

