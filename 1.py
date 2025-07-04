import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,f1_score,recall_score,precision_score,matthews_corrcoef,roc_auc_score,roc_curve,confusion_matrix,ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X = data.data
y = data.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = LogisticRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

tn,fp,fn,tp = confusion_matrix(y_pred,y_test).ravel()

acc = (tp + tn) / (tp + tn + fn + fp)
pre = (tp) / (tp + fp)
rec = (tp) / (tp + fn)
f1 = (2 * pre * rec) / (pre + rec)
spe = (tn) / (tn + fp)
npv = (tn)  /(tn + fn)
mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

print("Custom Metrics\n")
print(f"Accuracy: {acc:.2f}")
print(f"Precision: {pre:.2f}")
print(f"Recall: {rec:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Specificity: {spe:.2f}")
print(f"NPV: {npv:.2f}")
print(f"Mathews Correlation Coefficient: {mcc:.2f}")

print("Sklearn Metrics\n")
print(f"Accuracy: {accuracy_score(y_pred,y_test):.2f}")
print(f"Precision: {precision_score(y_pred,y_test):.2f}")
print(f"Recall: {recall_score(y_pred,y_test):.2f}")
print(f"F1 Score: {f1_score(y_pred,y_test):.2f}")
print(f"MCC: {matthews_corrcoef(y_pred,y_test):.2f}")


ConfusionMatrixDisplay.from_predictions(y_pred,y_test,cmap="Greens")
plt.grid(False)
plt.show()

tpr,fpr,_ = roc_curve(y_test,y_prob)
randome_probs = np.random.rand(len(y_test))
tpr_rand,fpr_rand,_ = roc_curve(y_test,randome_probs)

aur_model = roc_auc_score(y_test,y_prob)
aur_random = roc_auc_score(y_test,randome_probs)

plt.figure()
plt.plot(tpr,fpr,label=f"Model (AUC): {aur_model:.2f}")
plt.plot(tpr_rand,fpr_rand,"--",label = f"Random (AUC): {aur_random:.2f}")
plt.plot([0,1],[0,1],color = "grey")
plt.legend()
plt.grid(True)
plt.title('ROC Curve')
plt.xlabel('True Positive Rate')
plt.ylabel('False Positive Rate')
plt.show()
