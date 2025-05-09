#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This Script Created on Wed May  7 23:48:29 2025

@Author: G.Lakshman Teja ,Genomics Data Scientist

"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Sample gene expression data (rows: samples, columns: genes)
data = pd.DataFrame({
    'gene1': np.random.rand(100),
    'gene2': np.random.rand(100) * 2,
    'gene3': np.random.randint(0, 10, 100),
    'disease_status': np.random.choice(['healthy', 'disease1', 'disease2'], size=100)
})

X = data.drop('disease_status', axis=1)
y = data['disease_status']

# Encode the target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Initialize and train the XGBoost classifier
xgb_classifier = XGBClassifier(objective='multi:softmax', num_class=len(le.classes_), random_state=42)
xgb_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = xgb_classifier.predict(X_test)
y_pred_labels = le.inverse_transform(y_pred)
y_test_labels = le.inverse_transform(y_test)

# Evaluate performance
accuracy = accuracy_score(y_test_labels, y_pred_labels)
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_test_labels, y_pred_labels))

# Feature importance
feature_importances = pd.Series(xgb_classifier.feature_importances_, index=X.columns)
print("\nFeature Importances:")
print(feature_importances.sort_values(ascending=False))


#*************
'''
Predicting Breast Cancer from Genomic Variants

Dataset Overview:The study analyzed 7,498 patient records from Tartu University Hospital,
including 2,449 breast cancer cases. Variant Call Format (VCF) files were re-annotated using
the Variant Effect Predictor (VEP), extracting features such as:

IMPACT: Predicted effect of the variant.

QUAL: Quality score of the variant call.

DP: Read depth at the variant position.

QD: Quality by depth.

MAX_AF: Maximum allele frequency across populations.

These features served as inputs for the XGBoost model to predict cancer status.
'''

import pandas as pd

# Load the dataset
data = pd.read_csv('genomic_variants.csv')

# Features and target
X = data[['IMPACT', 'QUAL', 'DP', 'QD', 'MAX_AF']]
y = data['cancer_status']  # 1 for cancer, 0 for non-cancer

from sklearn.preprocessing import LabelEncoder

# Encode 'IMPACT' if it's categorical
if X['IMPACT'].dtype == 'object':
    le = LabelEncoder()
    X['IMPACT'] = le.fit_transform(X['IMPACT'])

from sklearn.model_selection import train_test_split

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

import xgboost as xgb

# Initialize the classifier
model = xgb.XGBClassifier(objective='binary:logistic',eval_metric='logloss',use_label_encoder=False,random_state=42)

# Train the model
model.fit(X_train, y_train)

from sklearn.metrics import classification_report, accuracy_score

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

import matplotlib.pyplot as plt
from xgboost import plot_importance

# Plot feature importance
plot_importance(model)
plt.title("Feature Importance")
plt.show()
