import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,recall_score,f1_score,roc_auc_score,ConfusionMatrixDisplay,precision_score,mean_squared_error,r2_score
)

data = load_breast_cancer()
X = data.data[:,[0]]
y = data.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = LogisticRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_pred,y_test)}")
print(f"Precision: {precision_score(y_pred,y_test)}")
print(f"Recall: {recall_score(y_pred,y_test)}")
print(f"F1 Score: {f1_score(y_pred,y_test)}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_pred,y_test))}")
print(f"R2: {r2_score(y_pred,y_test)}")

plt.figure(figsize=(8,5))
x_val = np.linspace(X.min(),X.max(),300).reshape(-1,1)
y_val = model.predict_proba(x_val)[:,1]
plt.scatter(X_test,y_test,c=y_pred,cmap='coolwarm',edgecolors='k')
plt.plot(x_val,y_val,label='Sigmoid Curve')
plt.xlabel("Mean Radius (Feature 1)")
plt.ylabel("Probability of Class = 1")
plt.title("Logistic Regression (Univariate)")
plt.legend()
plt.grid(True)
plt.show()

new_sample = [[10],[15],[20]]
prediction = model.predict(new_sample)
probability = model.predict_proba(new_sample)
for i,val in enumerate(new_sample):
    print(f"Sample: {val[0]} -> Prediction: {prediction[i]} -> Probabilities: {probability[i]}")