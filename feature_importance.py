import os
import pandas as pd
import numpy as np
import concurrent.futures

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from cleaner import class_mean_impute

data = '/Users/jackbuxton/Documents/Applied data science/Cwk/training'

def process_file(file_name):
    file_path = os.path.join(data, file_name)
    df = pd.read_csv(file_path, sep='|')
    df_filled = class_mean_impute(df)
    return df_filled

def main():
    psv_files = [f for f in os.listdir(data) if f.endswith('.psv')]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_file, psv_files))
    

    combined_df = pd.concat(results, ignore_index=True)
    print("Combined DataFrame shape:", combined_df.shape)
    print(combined_df.head())

    TARGET_COL = 'SepsisLabel'
    if TARGET_COL not in combined_df.columns:
        raise ValueError(f"'{TARGET_COL}' not found in combined_df columns.")

    y = combined_df[TARGET_COL].values
    X = combined_df.drop(columns=[TARGET_COL])
    X.fillna(X.mean(), inplace=True)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    y_pred_proba = rf.predict_proba(X_val)[:, 1]
    auc_score = roc_auc_score(y_val, y_pred_proba)
    print(f"Validation ROC AUC: {auc_score:.4f}")

    perm_result = permutation_importance(
        rf, X_val, y_val, n_repeats=5, random_state=42, scoring='roc_auc'
    )

    sorted_idx = perm_result.importances_mean.argsort()
    print("\nPermutation Feature Importances (by mean importance):")
    for idx in sorted_idx[::-1]:
        feature_name = X.columns[idx]
        importance_mean = perm_result.importances_mean[idx]
        print(f"{feature_name}: {importance_mean:.4f}")

if __name__ == "__main__":
    main()
