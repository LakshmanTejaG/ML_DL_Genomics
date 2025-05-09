#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This Script Created on Wed May  7 23:38:39 2025

@Author: G.Lakshman Teja ,Genomics Data Scientist

Python Libraries:

The scikit-learn library (sklearn.naive_bayes) provides several Naive Bayes implementations:

GaussianNB: Assumes features follow a Gaussian (normal) distribution. Less common for direct sequence or count-based genomic data.
MultinomialNB: Suitable for discrete data, such as word counts (for text) or k-mer counts in sequences.
ComplementNB: Often performs well on imbalanced text datasets and can be useful in some biological classification tasks.
BernoulliNB: Suitable for binary features (e.g., presence or absence of a specific k-mer).
CategoricalNB: Designed for explicitly categorical features.

How it works: In studies looking for associations between genetic variations (like SNPs - Single Nucleotide Polymorphisms) and diseases,
you can represent individuals by their genotypes at these SNP locations (e.g., AA, AG, GG). Naive Bayes can then be used to classify
individuals into disease and control groups based on their genotype profiles. The features are the discrete genotypes at each SNP.

"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

# Sample gene descriptions and functions
descriptions = [
    "encodes a protein involved in DNA repair",
    "catalyzes metabolic reactions in the cell",
    "functions as a transcriptional regulator",
    "involved in signal transduction pathways",
    "plays a role in cell growth and differentiation"
]
functions = ["DNA repair", "metabolism", "regulation", "signaling", "growth"]

# Feature extraction: Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(descriptions)
y = np.array(functions)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Multinomial Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Predict function for a new gene description
new_description = ["protein involved in energy production"]
new_features = vectorizer.transform(new_description)
predicted_function = nb_classifier.predict(new_features)
print(f"Predicted function: {predicted_function[0]}")

# Evaluate performance
y_pred = nb_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set: {accuracy}")