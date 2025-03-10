import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from scipy.stats import pointbiserialr, combine_pvalues
from cleaner import class_mean_impute

def compute_patient_corrs_and_pvals(grp, target_label='SepsisLabel'):
    r_values = {}
    p_values = {}
    for col in grp.columns:
        if col in [target_label, 'PatientID']:
            continue
        if grp[col].isna().all():
            r_values[col] = np.nan
            p_values[col] = np.nan
        else:
            r, p = pointbiserialr(grp[target_label], grp[col])
            r_values[col] = r
            p_values[col] = p
    return r_values, p_values

def main():
    data = '/Users/jackbuxton/Documents/Applied data science/Cwk/balanced_data'
    psv_files = [f for f in os.listdir(data) if f.endswith('.psv')]

    # Read in files and assign each a PatientID (using the filename)
    dfs = []
    for file in psv_files:
        file_path = os.path.join(data, file)
        df = pd.read_csv(file_path, sep='|')
        df['PatientID'] = file
        dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)

    print(combined_df.head())

    # Impute missing values via class_mean_impute function
    df_imputed = (
        combined_df
        .groupby('PatientID', as_index=False)
        .apply(class_mean_impute, include_groups=True)
    )

    print(df_imputed.head())

    # Calculate per-patient correlations and p-values
    patient_corrs = {}
    patient_pvals = {}
    for patient, grp in df_imputed.groupby('PatientID'):
        r_vals, p_vals = compute_patient_corrs_and_pvals(grp, target_label='SepsisLabel')
        patient_corrs[patient] = r_vals
        patient_pvals[patient] = p_vals

    corr_df = pd.DataFrame(patient_corrs).T
    pval_df = pd.DataFrame(patient_pvals).T

    mean_corr = corr_df.mean(axis=0)
    # Combine probability values using Fisher's method
    combined_p = {}
    for feature in pval_df.columns:
        valid_p = pval_df[feature].dropna()
        if len(valid_p) > 0:
            _, p_comb = combine_pvalues(valid_p, method='fisher')
            combined_p[feature] = p_comb
        else:
            combined_p[feature] = np.nan

    result_df = pd.DataFrame({
        'mean_correlation': mean_corr,
        'combined_p': pd.Series(combined_p)
    })
   
    # Avoid -inf in log10 by clipping very small p-values
    min_thresh = np.finfo(float).tiny
    result_df['neg_log10_p'] = -np.log10(result_df['combined_p'].clip(lower=min_thresh))
    result_df['abs_corr'] = result_df['mean_correlation'].abs()
    result_df.sort_values(by='abs_corr', ascending=False, inplace=True)

    print("\nMean Point-Biserial Correlation Across Patients with Combined P-values:")
    print(result_df.head(15))

    # ---------------------------------------------------
    # Volcano Plot (Full Range)
    # ---------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.scatter(result_df['mean_correlation'], result_df['neg_log10_p'], color='blue')
    plt.xlabel('Mean Point-Biserial Correlation')
    plt.ylabel('-log10(Combined P-value)')
    plt.title('Volcano Plot: Correlation vs. Statistical Significance')
    
    # Draw significance thresholds
    plt.axhline(y=-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
    plt.axvline(x=0, color='grey', linestyle='--')

    # Annotate features on the plot
    for feature, row in result_df.iterrows():
        plt.text(row['mean_correlation'], row['neg_log10_p'], feature, 
                 fontsize=8, ha='center', va='bottom')
    
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------
    # Volcano Plot (Zoomed-In) around correlation=0
    # ---------------------------------------------------
    # You can adjust these x-limits and y-limits based on your data distribution
    zoom_xmin, zoom_xmax = -0.1, 0.1

    plt.figure(figsize=(10, 6))
    plt.scatter(result_df['mean_correlation'], result_df['neg_log10_p'], color='blue')
    plt.xlabel('Mean Point-Biserial Correlation (Zoomed)')
    plt.ylabel('-log10(Combined P-value)')
    plt.title('Volcano Plot (Zoomed Near Correlation=0)')

    # Apply the same thresholds
    plt.axhline(y=-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
    plt.axvline(x=0, color='grey', linestyle='--')

    # Adjust x-limits (and y-limits if desired)
    plt.xlim(zoom_xmin, zoom_xmax)
    # Example y-limit if you want to zoom in on smaller p-values:
    # plt.ylim(0, 3)  # adjust as needed

    # Annotate features (optional, can get messy if many points overlap)
    for feature, row in result_df.iterrows():
        if zoom_xmin <= row['mean_correlation'] <= zoom_xmax:
            plt.text(row['mean_correlation'], row['neg_log10_p'], feature,
                     fontsize=8, ha='center', va='bottom')

    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
