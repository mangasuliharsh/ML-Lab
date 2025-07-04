import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

# Load the Iris dataset
data = load_iris()
X = data.data
y = data.target
target_names = data.target_names
feature_names = data.feature_names

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree using ID3 (entropy)
model = DecisionTreeClassifier(criterion="entropy", random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluation
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.2f}")
print(f"Classification Report:\n{classification_report(y_test, y_pred, target_names=target_names)}")

# Plot the decision tree
plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=feature_names, class_names=target_names.tolist(), filled=True)
plt.title("Decision Tree Classifier (ID3 - Entropy)")
plt.show()


new_sample = [[5,6,7,2]]
prediction = model.predict(new_sample)
print(f"Sample: {new_sample} -> Prediction: {target_names[prediction]}")
