#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This Script Created on Wed May  7 20:03:27 2025

@Author: G.Lakshman Teja ,Genomics Data Scientist

"""
import pandas as pd
import numpy as np
from tabulate import tabulate

# 1. Create synthetic SNP data
np.random.seed(42)
n_samples = 200
annotation_data = pd.DataFrame({
    'conservation_score': np.random.normal(loc=0.5, scale=0.2, size=n_samples),
    'amino_acid_change': np.random.choice(['A>G', 'C>T', 'G>A', 'T>C', 'A>C', 'G>T'], size=n_samples),
    'distance_to_exon_boundary': np.random.randint(0, 100, size=n_samples),
    'functional_domain': np.random.choice(['kinase', 'transcription_factor', 'binding', 'other', None], size=n_samples),
    'sift_score': np.random.uniform(0, 1, size=n_samples),
    'polyphen_score': np.random.uniform(0, 1, size=n_samples),
    'allele_frequency': np.random.beta(1, 10, size=n_samples),
    'pathogenicity': np.random.choice(['pathogenic', 'benign'], size=n_samples, p=[0.4, 0.6]) # Imbalanced classes
})

# Introduce some relationships (for demonstration purposes)
annotation_data.loc[annotation_data['conservation_score'] > 0.7, 'pathogenicity'] = 'pathogenic'
annotation_data.loc[(annotation_data['distance_to_exon_boundary'] < 10) & (annotation_data['pathogenicity'] == 'benign'), 'pathogenicity'] = 'pathogenic'
annotation_data.loc[(annotation_data['amino_acid_change'].isin(['A>C', 'G>T'])) & (annotation_data['pathogenicity'] == 'benign'), 'pathogenicity'] = 'pathogenic'
annotation_data.fillna({'functional_domain': 'unknown'}, inplace=True)

print(tabulate(annotation_data, headers='keys',tablefmt='psql',showindex=False))

# Writing output as Excel file
annotation_data.to_excel('TSHC_Variants_Annotation_Data.xlsx', engine='xlsxwriter',sheet_name='Annot_SNP', header=True,index=False)