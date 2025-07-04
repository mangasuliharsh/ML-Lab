import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

data = load_diabetes()

X,y = data.data,(data.target > 140).astype(int)

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def KNN(X_train,y_train,X_test,K,metrics):
    preds = []
    for x in X_test:
        distance = np.linalg.norm(X_train-x,axis=1) if metrics == 'euclidean' else np.sum(np.abs(X_train-x),axis=1)
        idx = np.argsort(distance)[:K]
        votes = y_train[idx]
        preds.append(np.bincount(votes).argmax())
    return np.array(preds)

for K in [3,5,7,9,11,13]:
    for metrics in ['euclidean','manhattan']:
        y_pred = KNN(X_train,y_train,X_test,K,metrics)
        acc = accuracy_score(y_test,y_pred)
        print(f"K = {K}   | Metrics = {metrics}   | Accuracy = {acc}")