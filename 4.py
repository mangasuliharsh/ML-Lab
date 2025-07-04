import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,r2_score
)

data = load_diabetes()
X = data.data[:,np.newaxis,2]
y = data.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

model = LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)


rmse = np.sqrt(mean_squared_error(y_pred,y_test))
r2 = r2_score(y_pred,y_test)

print(f"Root Mean Squared Error: {rmse}")
print(f"R2 Score: {r2}")

plt.scatter(X_test,y_test,color='blue',label = 'Actual')
plt.plot(X_test,y_pred,color='red',label='Regression')
plt.xlabel('BMI')
plt.ylabel('Desceased Rate')
plt.title('Linear Regression')
plt.legend()
plt.grid(True)
plt.show()
