import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,recall_score,f1_score,roc_auc_score,ConfusionMatrixDisplay,precision_score
)

data = load_breast_cancer()
X = data.data
y = data.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

n_estimators_list = [1,5,10,20,50,100,150,200]

results = {
    'n_estimator':[],
    'accuracy':[],
    'recall':[],
    'precision':[],
    'f1':[],
    'roc_score':[]
}

for n in n_estimators_list:
    model = RandomForestClassifier(n_estimators=n,random_state=42)
    model.fit(X_train,y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
    
    results['n_estimator'].append(n)
    results['accuracy'].append(accuracy_score(y_pred,y_test))
    results['f1'].append(f1_score(y_pred,y_test))
    results['precision'].append(precision_score(y_pred,y_test))
    results['recall'].append(recall_score(y_pred,y_test))
    results['roc_score'].append(roc_auc_score(y_test,y_prob))
    
plt.plot(results['n_estimator'],results['accuracy'],color='red',label='Accuracy')
plt.plot(results['n_estimator'],results['f1'],color='blue',label='F1 Score')
plt.plot(results['n_estimator'],results['precision'],color='yellow',label='Precision')
plt.plot(results['n_estimator'],results['recall'],color='orange',label='Recall Score')
plt.plot(results['n_estimator'],results['roc_score'],color='green',label='ROC Score')
plt.title('Random Forest Classifier')
plt.ylabel('Scores')
plt.xlabel('Estimators')
plt.legend()
plt.grid(True)
plt.show()