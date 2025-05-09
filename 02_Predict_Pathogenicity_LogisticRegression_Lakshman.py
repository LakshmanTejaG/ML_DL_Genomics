#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This Script Created on Wed May  7 21:28:29 2025

@Author: G.Lakshman Teja ,Genomics Data Scientist

"""
# Importing python required packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Create synthetic SNP data
np.random.seed(42)
n_samples = 200
data = pd.DataFrame({
    'conservation_score': np.random.normal(loc=0.5, scale=0.2, size=n_samples),
    'amino_acid_change': np.random.choice(['A>G', 'C>T', 'G>A', 'T>C', 'A>C', 'G>T'], size=n_samples),
    'distance_to_exon_boundary': np.random.randint(0, 100, size=n_samples),
    'functional_domain': np.random.choice(['kinase', 'transcription_factor', 'binding', 'receptor','other', None], size=n_samples),
    'pathogenicity': np.random.choice(['pathogenic', 'benign'], size=n_samples, p=[0.4, 0.6]) # Imbalanced classes
})

# Introduce some relationships (for demonstration purposes)
data.loc[data['conservation_score'] > 0.7, 'pathogenicity'] = 'pathogenic'
data.loc[(data['distance_to_exon_boundary'] < 10) & (data['pathogenicity'] == 'benign'), 'pathogenicity'] = 'pathogenic'
data.loc[(data['amino_acid_change'].isin(['A>C', 'G>T'])) & (data['pathogenicity'] == 'benign'), 'pathogenicity'] = 'pathogenic'
data.fillna({'functional_domain': 'unknown'}, inplace=True)

# 2. Data Preprocessing
# Encode the target variable
label_encoder = LabelEncoder()
data['pathogenicity_encoded'] = label_encoder.fit_transform(data['pathogenicity']) # benign: 0, pathogenic: 1

# Handle categorical features
data = pd.get_dummies(data, columns=['amino_acid_change', 'functional_domain'], drop_first=True)

# Separate features (X) and target (y)
X = data.drop(['pathogenicity', 'pathogenicity_encoded'], axis=1)
y = data['pathogenicity_encoded']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y) # Stratify to maintain class proportions

# Scale numerical features
numerical_features = ['conservation_score', 'distance_to_exon_boundary']
scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# 3. Model Training
model = LogisticRegression(random_state=42, solver='liblinear') # 'liblinear' is good for small to medium datasets
model.fit(X_train, y_train)

# 4. Model Evaluation
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Compute confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# Calculate metrics
precision = tp / (tp + fp)
recall = tp / (tp + fn)  # same as sensitivity
accuracy = (tp + tn) / (tp + tn + fp + fn)

print(f"Precision: {precision:.2f}")
print(f"Recall (Sensitivity): {recall:.2f}")
print(f"Accuracy: {accuracy:.2f}")
