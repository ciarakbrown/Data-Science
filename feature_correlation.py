import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, levene
from scipy.stats import pointbiserialr
from sklearn.feature_selection import mutual_info_classif
from class_balance import load_data
import matplotlib as mpl

mpl.rcParams['text.usetex'] = True

mpl.rcParams.update({
    'font.size': 18,         # general font size
    'axes.titlesize': 20,    # title size for axes
    'axes.labelsize': 18,    # x and y labels
    'xtick.labelsize': 18,   # x tick labels
    'ytick.labelsize': 18,   # y tick labels
    'legend.fontsize': 18    # legend font size
})

def remove_outliers(df):
    demographic_columns = ['HospAdmTime', 'Gender', 'Age', 'Unit1', 'Unit2']
    features = [col for col in df.columns if col not in (['SepsisLabel', 'patient_id'] + demographic_columns)]
    
    melted_df = df.melt(value_vars=features, var_name='Feature', value_name='Value')
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=melted_df, x='Feature', y='Value', palette='Set3')
    plt.title('Original Distribution of Continuous Features')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('figs/boxplot_original.png', dpi=300)
    plt.show()
    
    df_clean = df.copy()
    for col in features:
        q1 = df_clean[col].quantile(0.25)
        q3 = df_clean[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    melted_df_clean = df_clean.melt(value_vars=features, var_name='Feature', value_name='Value')
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=melted_df_clean, x='Feature', y='Value', palette='Set3')
    plt.title('Distribution of Continuous Features After Outlier Removal')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('figs/boxplot_cleaned.png', dpi=300)
    plt.show()
    
    return df_clean

def point_biserial_corr(df):
    demographic_columns = ['HospAdmTime','Gender', 'Age', 'Unit1', 'Unit2']
    patient_corrs = {}
    patient_pvals = {}
    for patient, group in df.groupby('patient_id'):
         group = group.drop(columns=demographic_columns, errors='ignore')
         r_values = {}
         p_values = {}
         for col in group.columns:
             if col in ['SepsisLabel', 'patient_id']:
                 continue
             subset = group[['SepsisLabel', col]].dropna()
             if len(subset) < 2:
                 r_values[col] = np.nan
                 p_values[col] = np.nan
             else:
                 r, p = pointbiserialr(subset['SepsisLabel'], subset[col])
                 r_values[col] = r
                 p_values[col] = p
         patient_corrs[patient] = r_values
         patient_pvals[patient] = p_values
    corr_df = pd.DataFrame(patient_corrs).T 
    pval_df = pd.DataFrame(patient_pvals).T
    return corr_df, pval_df

from scipy.stats import spearmanr

def spearman_corr(df):
    demographic_columns = ['HospAdmTime', 'Gender', 'Age', 'Unit1', 'Unit2']
    
    patient_corrs = {}
    patient_pvals = {}
    
    for patient, group in df.groupby('patient_id'):
         group = group.drop(columns=demographic_columns, errors='ignore')
         r_values = {}
         p_values = {}
         for col in group.columns:
             if col in ['SepsisLabel', 'patient_id']:
                 continue
             subset = group[['SepsisLabel', col]].dropna()
             if len(subset) < 2:
                 r_values[col] = np.nan
                 p_values[col] = np.nan
             else:
                 r, p = spearmanr(subset['SepsisLabel'], subset[col])
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
         features = group.drop(columns=['SepsisLabel', 'patient_id', 'Unit1', 'Unit2'], errors='ignore')
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
    plt.title('Patient-Level Correlations by Feature')
    plt.xlabel('Feature')
    plt.ylabel('Point-biserial Correlation')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('figs/biserial_corr.png', dpi=300)
    plt.show()

def plot_significance(pval_df):
    pval_long = (
        pval_df
        .reset_index(names='patient_id')   
        .melt(id_vars='patient_id',
              var_name='Feature',
              value_name='p_value')     
    )
    
    small_p = 1e-20
    pval_long['p_value_clipped'] = pval_long['p_value'].clip(lower=small_p)
    pval_long['neg_log10_p'] = -np.log10(pval_long['p_value_clipped'])
    
    plt.figure(figsize=(12, 6))
    sns.violinplot(
        data=pval_long,
        x='Feature',
        y='neg_log10_p',
        color='skyblue',
        cut=0  
    )
    plt.axhline(y=-np.log10(0.01), color='red', linestyle='--', label='p=0.05')
    plt.title('Patient-Level p-values by Feature')
    plt.xlabel('Feature')
    plt.ylabel('-log10(p-value)')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('figs/biserial_significance.png', dpi=300)
    plt.show()


def plot_mi(mi_df):
    plt.figure(figsize=(10,6))
    sns.barplot(x=mi_df.index, y=mi_df.values, color='blue')
    plt.ylabel('Mean Mutual Information')
    plt.xlabel('Feature')
    plt.title('Mean Mutual Information for Each Feature Across Patients')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('figs/plot_mi.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    data_path = '/Users/jackbuxton/Documents/Applied data science/Cwk/low_dim_balanced_data'
    combined_df = load_data(data_path)
    # clean_df = remove_outliers(combined_df)
    
    pb_corr_df, pb_pval_df = point_biserial_corr(combined_df)
    plot_corr(pb_corr_df)
    plot_significance(pb_pval_df)
    
    # sp_corr_df, sp_pval_df = spearman_corr(combined_df)
    # plot_corr(sp_corr_df)
    # plot_significance(sp_pval_df)