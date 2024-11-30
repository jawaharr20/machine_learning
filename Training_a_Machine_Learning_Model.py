import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
# Sample dataset (using a toy dataset for demonstration)
data = {
'age': [25, 30, 35, 20, 40, 45, 22, 38, 50, 60],
'income': [50000, 70000, 60000, 30000, 80000, 90000, 40000, 120000,
100000, 150000],
'purchased': [0, 1, 1, 0, 1, 1, 0, 1, 1, 1]
}
df = pd.DataFrame(data)
# Separate features and target variable
X = df[['age', 'income']]
y = df['purchased']
# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)# Initialize Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
# Train the model
clf.fit(X_train, y_train)
# Predict on the test set
y_pred = clf.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=1))