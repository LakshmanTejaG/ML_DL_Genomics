#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This Script Created on Wed May  7 18:52:25 2025

@Author: G.Lakshman Teja ,Genomics Data Scientist

Use Case: Predict Tumor Mutational Burden (TMB) from Variants Count per Gene.
Objective: Use linear regression to see if gene-level variant load (e.g., TP53, BRCA1) predicts TMB.

"""
# Importing Libraries
import pandas as pd
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Importing the data into dataframe
tso500_data = pd.read_csv('data/tso500_variant_tmb_data.csv',low_memory=False,index_col=None)
print(tabulate(tso500_data,headers='keys',tablefmt='psql',showindex=False))

# Features and target
X = tso500_data[['TP53_variants','BRCA1_variants']]
y = tso500_data[['TMB']]
print(X)
print(y)

# Splitting the data into training and test sets
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# Fitting the data into model
model = LinearRegression()
model.fit(X_train, y_train)

# Model Predictions
y_pred = model.predict(X_test)
print(y_pred)

# Model Evaluation Metrics
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)
print("R² Score:", r2_score(y_test, y_pred))

# Plot actual vs predicted TMB
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual TMB')
plt.ylabel('Predicted TMB')
plt.title('Actual vs Predicted')
plt.grid(True)
plt.show()


