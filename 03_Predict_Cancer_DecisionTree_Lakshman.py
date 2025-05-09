#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This Script Created on Wed May  7 22:19:03 2025
@Author: G.Lakshman Teja ,Genomics Data Scientist

"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Step 1: Load the gene expression dataset
# Example: 'gene_expression.csv' where each row represents a tissue sample and each column represents gene expression levels
# The target column 'label' contains 1 (Cancer) or 0 (Non-Cancer)

# Replace 'gene_expression.csv' with the path to your actual data file
data = pd.read_csv('data/gene_expression.csv')

# Step 2: Split data into features (X) and target labels (y)
X = data.drop(columns=['label'])  # Drop the target column 'label' to get the features (gene expressions)
y = data['label']  # The target column is 'label', which indicates whether the sample is cancerous or non-cancerous

# Step 3: Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Initialize and train the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Step 5: Make predictions on the test data
y_pred = clf.predict(X_test)

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 7: Feature importance to identify important genes
feature_importance = clf.feature_importances_
important_genes = pd.DataFrame({
    'Gene': X.columns,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

print("\nTop 10 Important Genes:")
print(important_genes.head(10))

# Step 8: Visualize the decision tree
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=X.columns, class_names=["Non-Cancer", "Cancer"], filled=True)
plt.title("Decision Tree for Cancer Classification")
plt.show()
