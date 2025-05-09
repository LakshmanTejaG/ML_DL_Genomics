#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This Script Created on Wed May  7 22:25:20 2025

@Author: G.Lakshman Teja ,Genomics Data Scientist

"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Assume 'gene_expression.csv' contains gene expression data with a 'disease_status' column
data = pd.read_csv('gene_expression.csv', index_col=0)
X = data.drop('disease_status', axis=1)
y = data['disease_status']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the RandomForest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Feature Importance
feature_importances = pd.Series(rf_classifier.feature_importances_, index=X.columns)
top_n_features = feature_importances.nlargest(10)
print("\nTop 10 Important Features:\n", top_n_features)