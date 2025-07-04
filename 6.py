import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
)

# Load the Iris dataset
data = load_iris()
X = data.data
y = data.target
target_names = data.target_names

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Plot the confusion matrix with class labels
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Greens", display_labels=target_names)
plt.title("Confusion Matrix")
plt.grid(False)
plt.show()

# Print evaluation metrics
print(f"Classification Report: \n{classification_report(y_test, y_pred, target_names=target_names.tolist())}")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Predict on new samples
new_sample = [[10, 6, 7, 3],
              [2, 3, 4, 1]]
predictions = model.predict(new_sample)

# Display predictions with class names
for i, val in enumerate(predictions):
    print(f"Sample {i+1} -> {target_names[val]}")
