import numpy as np 
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

data = load_iris()

X = data.data
y = data.target

PCA = PCA(n_components=2)
x_pca = PCA.fit_transform(X)

K = [2,3,4]
plt.figure(figsize=(15,4))

for i,k in enumerate(K):
    kmeans = KMeans(n_clusters=k,random_state=42)
    clusters = kmeans.fit_predict(X)
    
    centriod_pca = PCA.transform(kmeans.cluster_centers_)
    
    plt.subplot(1,3,i+1)
    plt.scatter(x_pca[:,0],x_pca[:,1],c=clusters,cmap='viridis',label='Data')
    plt.scatter(centriod_pca[:,0],centriod_pca[:,1],s=200,marker='X',c='red',label = 'Centroids')
    plt.title(f"K {k}")
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.legend()
plt.tight_layout()
plt.suptitle('K-Means Clustering', fontsize=16, y=1.05)
plt.show()