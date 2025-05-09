#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This Script Created on Wed May  7 23:21:27 2025

@Author: G.Lakshman Teja ,Genomics Data Scientist

How it works: Genes with similar expression patterns (measured through techniques like microarrays or RNA-seq)
are often involved in similar biological processes. We can train a K-NN classifier using genes with known
functions as training data. The features would be the expression levels of the genes across different
conditions. For a new gene with unknown function, the algorithm finds the 'K' genes with the most similar
expression profiles and predicts the function based on the majority function among its neighbors.

"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Sample gene expression data (rows: genes, columns: conditions)
X = np.array([[1, 2, 3, 4],
              [1.5, 2.2, 3.1, 4.2],
              [5, 6, 7, 8],
              [5.1, 6.1, 7.2, 8.1],
              [2, 3, 1, 2]])

# Corresponding known functions
y = np.array(['metabolism', 'metabolism', 'signaling', 'signaling', 'DNA repair'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the K-NN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predict function for a new gene expression profile
new_gene_expression = np.array([[1.2, 2.1, 3.2, 4.1]])
predicted_function = knn.predict(new_gene_expression)
print(f"Predicted function: {predicted_function[0]}")

# Evaluate performance on the test set
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set: {accuracy}")