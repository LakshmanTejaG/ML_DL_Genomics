#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This Script Created on Thu May  8 15:30:48 2025

@Author: G.Lakshman Teja ,Genomics Data Scientist

"""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Sample patient mutation data (rows are patients, columns are genes)
mutation_data = pd.DataFrame({
    'GeneA': [1, 0, 1, 0, 1, 0],
    'GeneB': [0, 1, 0, 1, 0, 1],
    'GeneC': [1, 1, 0, 0, 1, 1],
    'GeneD': [0, 0, 1, 1, 0, 0]
}, index=['Patient1', 'Patient2', 'Patient3', 'Patient4', 'Patient5', 'Patient6'])

# Standardize the data if the features have different scales
scaler = StandardScaler()
scaled_mutations = scaler.fit_transform(mutation_data)

# Determine the optimal number of clusters (Elbow method)
inertia = []
k_range = range(1, scaled_mutations.shape[0]+1)
for k in k_range:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(scaled_mutations)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.xticks(k_range)
plt.grid(True)
plt.show()

# Choose the number of patient subtypes (e.g., 2)
n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
patient_clusters = kmeans.fit_predict(scaled_mutations)

# Add cluster labels to the DataFrame
mutation_data['Subtype'] = patient_clusters

print("\nPatient Subtypes based on Mutations:")
print(mutation_data)

# Visualize the subtypes using PCA for dimensionality reduction
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_mutations)
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['Subtype'] = patient_clusters

plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Subtype', data=pca_df, palette='Set1')
plt.title(f'Patient Subtypes (k={n_clusters})')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()