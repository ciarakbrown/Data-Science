import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pointbiserialr
from sklearn.feature_selection import mutual_info_classif
from class_balance import load_data
from cleaner import class_mean_impute
from scipy.stats import combine_pvalues

def point_biserial_corr(df):
    r_values = {}
    p_values = {}
    for col in df.columns:
        if col in ['SepsisLabel', 'patient_id']:
            continue
        if df[col].isna().all():
            r_values[col] = np.nan
            p_values[col] = np.nan
        else:
            r, p = pointbiserialr(df['SepsisLabel'], df[col])
            r_values[col] = r
            p_values[col] = p
    return r_values, p_values

# Not finished yet
def mutual_info(df, n_features=40):
    """
    Performs mutual information feature selection on the dataset.
    Returns the top `n_features` most informative features.
    """
    features = df.drop(columns=['SepsisLabel', 'patient_id'], errors='ignore')
    target = df['SepsisLabel']
    
    mi_scores = mutual_info_classif(features.fillna(0), target, discrete_features=False)
    mi_df = pd.DataFrame({'Feature': features.columns, 'Mutual_Information': mi_scores})
    mi_df.sort_values(by='Mutual_Information', ascending=False, inplace=True)
    
    print("Top Features Based on Mutual Information:")
    print(mi_df.head(n_features))
    return mi_df.head(n_features)

def plot_volcano(df, title, zoom_xmin=None, zoom_xmax=None):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['mean_correlation'], df['neg_log10_p'], color='blue')
    plt.xlabel('Mean Point-Biserial Correlation')
    plt.ylabel('-log10(Combined P-value)')
    plt.title(title)
    plt.axhline(y=-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
    plt.axvline(x=0, color='grey', linestyle='--')
    
    # Annotate features
    for feature, row in df.iterrows():
        plt.text(row['mean_correlation'], row['neg_log10_p'], feature, fontsize=8, ha='center', va='bottom')
    
    if zoom_xmin is not None and zoom_xmax is not None:
        plt.xlim(zoom_xmin, zoom_xmax)
    
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data_path = '/Users/jackbuxton/Documents/Applied data science/Cwk/balanced_data'
    combined_df = load_data(data_path)
    df_imputed = (
        combined_df
        .groupby('patient_id', as_index=False)
        .apply(class_mean_impute, include_groups=True)
    )

    mutual_information = mutual_info(df_imputed, n_features=40)

    patient_corrs = {}
    patient_pvals = {}
    for patient, grp in df_imputed.groupby('patient_id'):
        r_vals, p_vals = point_biserial_corr(grp)
        patient_corrs[patient] = r_vals
        patient_pvals[patient] = p_vals

    corr_df = pd.DataFrame(patient_corrs).T
    pval_df = pd.DataFrame(patient_pvals).T
    mean_corr = corr_df.mean(axis=0)

    # Combine p-values using Fisher's method
    combined_p = {
        feature: combine_pvalues(pval_df[feature].dropna(), method='fisher')[1] if not pval_df[feature].dropna().empty else np.nan
        for feature in pval_df.columns
    }

    result_df = pd.DataFrame({
        'mean_correlation': mean_corr,
        'combined_p': pd.Series(combined_p)
    })

    # Avoid -inf in log10 by clipping very small p-values
    min_thresh = np.finfo(float).tiny
    result_df['neg_log10_p'] = -np.log10(result_df['combined_p'].clip(lower=min_thresh))
    result_df['abs_corr'] = result_df['mean_correlation'].abs()
    result_df.sort_values(by='abs_corr', ascending=False, inplace=True)

    plot_volcano(result_df, 'Volcano Plot: Correlation vs. Statistical Significance')
    plot_volcano(result_df, 'Volcano Plot (Zoomed Near Correlation=0)', zoom_xmin=-0.1, zoom_xmax=0.1)

    # Compute and plot the feature correlation matrix
    feature_corr_matrix = df_imputed.drop(columns=['SepsisLabel', 'patient_id'], errors='ignore').corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(feature_corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=mutual_information['Mutual_Information'], y=mutual_information['Feature'], palette='viridis')
    plt.xlabel('Mutual Information Score')
    plt.ylabel('Feature')
    plt.title('Top Features Based on Mutual Information')
    plt.show()
