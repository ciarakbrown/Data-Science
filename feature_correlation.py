import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pointbiserialr
from sklearn.feature_selection import mutual_info_classif
from class_balance import load_data

def point_biserial_corr(df):
    demographic_columns = ['HospAdmTime','Gender', 'Age_18-44', 'Age_45-59', 
                             'Age_60-64', 'Age_65-74', 'Age_75-79', 'Age_80-89']
    patient_corrs = {}
    patient_pvals = {}
    for patient, group in df.groupby('patient_id'):
         group = group.drop(columns=demographic_columns, errors='ignore')
         r_values = {}
         p_values = {}
         for col in group.columns:
             if col in ['SepsisLabel', 'patient_id']:
                 continue
             r, p = pointbiserialr(group['SepsisLabel'], group[col])
             r_values[col] = r
             p_values[col] = p
         patient_corrs[patient] = r_values
         patient_pvals[patient] = p_values
    corr_df = pd.DataFrame(patient_corrs).T 
    pval_df = pd.DataFrame(patient_pvals).T
    return corr_df, pval_df

def mutual_info(df):
    mi_dict = {}
    for patient, group in df.groupby('patient_id'):
         features = group.drop(columns=['SepsisLabel', 'patient_id'], errors='ignore')
         target = group['SepsisLabel']
         mi_scores = mutual_info_classif(features, target, discrete_features=False)
         mi_dict[patient] = dict(zip(features.columns, mi_scores))
    mi_df = pd.DataFrame(mi_dict).T
    return mi_df.mean().sort_values(ascending=False)

def plot_corr(corr_df):
    corr_long = (
        corr_df
        .reset_index(names='patient_id')
        .melt(id_vars='patient_id', var_name='Feature', value_name='Correlation')
    )

    plt.figure(figsize=(12, 6))
    sns.violinplot(
        data=corr_long,
        x='Feature',
        y='Correlation',
        color='skyblue',
        cut=0
    )

    plt.axhline(y=0, color='gray', linestyle='--')
    plt.title('Violin Plot: Patient-Level Correlations by Feature')
    plt.xlabel('Feature')
    plt.ylabel('Point-Biserial Correlation')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_significance(pval_df):
    pval_long = (
        pval_df
        .reset_index(names='patient_id')   
        .melt(id_vars='patient_id',
              var_name='Feature',
              value_name='p_value')     
    )
    tiny = 1e-300
    pval_long['neg_log10_p'] = -np.log10(pval_long['p_value'].clip(lower=tiny))
    
    plt.figure(figsize=(12, 6))
    sns.violinplot(
        data=pval_long,
        x='Feature',
        y='neg_log10_p',
        color='skyblue',
        cut=0  
    )
    
    plt.axhline(y=-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
    plt.title('Violin Plot: Patient-Level p-values by Feature')
    plt.xlabel('Feature')
    plt.ylabel('-log10(p-value)')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_mi(mi_df):
    plt.figure(figsize=(10,6))
    sns.barplot(x=mi_df.index, y=mi_df.values, color='blue')
    plt.ylabel('Mean Mutual Information')
    plt.title('Mean Mutual Information for Each Feature Across Patients')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data_path = '/Users/jackbuxton/Documents/Applied data science/Cwk/cleaned_dataset'
    combined_df = load_data(data_path)

    mi_df = mutual_info(combined_df)
    plot_mi(mi_df)

    corr_df, pval_df = point_biserial_corr(combined_df)
    plot_corr(corr_df)
    plot_significance(pval_df)