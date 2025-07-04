import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,recall_score,precision_score,f1_score,roc_curve,ConfusionMatrixDisplay )
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
X = data.data
y = data.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

ada = AdaBoostClassifier(n_estimators=100,random_state=42)
xgb = XGBClassifier(use_label_encoder = False,eval_metrics = 'logloss',random_state = 42)

ada.fit(X_train,y_train)
xgb.fit(X_train,y_train)

y_pred_ada = ada.predict(X_test)
y_pred_xgb = xgb.predict(X_test)


def evaluate(y_test,y_pred,model):
    print(f"{model}")
    print(f"Accuracy: {accuracy_score(y_test,y_pred)}")
    print(f"recall: {recall_score(y_test,y_pred)}")
    print(f"Precision: {precision_score(y_test,y_pred)}")
    print(f"F1: {f1_score(y_test,y_pred)}")
    ConfusionMatrixDisplay.from_predictions(y_test,y_pred)
    plt.title(f"Confusion Matrix: {model} ")
    plt.show()
    
evaluate(y_test,y_pred_ada,'AdaBoost')
evaluate(y_test,y_pred_xgb,'XGBoost')

tpr_ada,fpr_ada,_ = roc_curve(y_test,ada.predict_proba(X_test)[:,1])
tpr_xgb,fpr_xgb,_ = roc_curve(y_test,xgb.predict_proba(X_test)[:,1])


plt.plot(tpr_ada,fpr_ada,color='red',label='Adaboost')
plt.plot(tpr_xgb,fpr_xgb,color='green',label='XGBoost')
plt.plot([0,1],[0,1],'k--')
plt.legend()
plt.show()