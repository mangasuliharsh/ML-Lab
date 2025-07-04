import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler

data = load_iris()

X = StandardScaler().fit_transform(data.data)

#AGNES
agnes_link = linkage(X,method='ward')
plt.figure(figsize=(8,4))
dendrogram(agnes_link,truncate_mode='level',p=5)
plt.title("AGNES DENDROGRAM")
plt.show()

agnes_labels = AgglomerativeClustering(n_clusters=3).fit_predict(X)
sns.scatterplot(x=X[:,0],y=X[:,1],hue=agnes_labels,palette="Set1")
plt.title("AGNES Clustering")
plt.show()

#DIANA
diana_link = linkage(pdist(X),method='average')
plt.figure(figsize=(8,4))
dendrogram(diana_link,truncate_mode = 'level',p=5)
plt.title("DIANA DENDROGRAM")
plt.show()

dinana_labels = fcluster(diana_link,3,criterion='maxclust')
sns.scatterplot(x=X[:,0],y=X[:,1],hue=dinana_labels,palette="Set1")
plt.title("DIANA Clustering")
plt.show()