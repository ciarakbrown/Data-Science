import os
import zipfile
import glob
import pandas as pd
import numpy as np
import tempfile
import shutil
import argparse
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from skopt import BayesSearchCV

# Add utils path for evaluate_sepsis_score
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from evaluate_sepsis_score import evaluate_sepsis_score

# Global parameters
WINDOW_SIZE_DEFAULT = 6
OFFSET_RANGE = range(-12, 0)

# Sliding window feature extractor for a given offset
def get_sliding_windows(df_all, offset, window_size):
    rows = []
    for patient_id, group in df_all.groupby("PatientID"):
        group = group.reset_index(drop=True)
        sepsis_indices = group.index[group["SepsisLabel"] == 1].tolist()

        if sepsis_indices:
            diagnosis_time = sepsis_indices[0]
            snapshot_time = diagnosis_time + offset
            label = 1
        else:
            if len(group) + offset < 0 or len(group) < window_size:
                continue
            snapshot_time = len(group) + offset
            label = 0

        start = snapshot_time - window_size + 1
        end = snapshot_time + 1
        if start < 0 or end > len(group):
            continue

        window = group.iloc[start:end].copy()
        features = window.mean(numeric_only=True)
        features["SepsisLabel"] = label
        features["PatientID"] = patient_id
        features["Hour"] = offset
        rows.append(features)
    return pd.DataFrame(rows)

# Write temp files and call evaluate_sepsis_score
def compute_physionet_metrics(y_true, y_pred, y_prob, patient_ids):
    label_dir = tempfile.mkdtemp()
    pred_dir = tempfile.mkdtemp()
    try:
        for i, pid in enumerate(patient_ids):
            df_label = pd.DataFrame({"SepsisLabel": [y_true[i]] * 10})
            df_pred = pd.DataFrame({
                "PredictedProbability": [y_prob[i]] * 10,
                "PredictedLabel": [y_pred[i]] * 10
            })
            df_label.to_csv(os.path.join(label_dir, f"{pid}.psv"), sep='|', index=False)
            df_pred.to_csv(os.path.join(pred_dir, f"{pid}.psv"), sep='|', index=False)
        auroc, auprc, accuracy, f1, utility = evaluate_sepsis_score(label_dir, pred_dir)
    finally:
        shutil.rmtree(label_dir)
        shutil.rmtree(pred_dir)
    return {
        'Utility': utility,
        'AUROC': auroc,
        'AUPRC': auprc,
        'Accuracy': accuracy,
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1
    }

# Main pipeline: generate features, train model, predict and evaluate
def run_pipeline(zip_path, window_size, threshold, output_csv):
    # Unzip dataset
    if not os.path.exists(zip_path):
        print("Invalid dataset path.")
        return
    extract_dir = "./sepsis_dataset"
    if not os.path.exists(extract_dir):
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_dir)

    # Load all patient data
    data_dir = os.path.join(extract_dir, "cleaned_dataset")
    files = glob.glob(os.path.join(data_dir, "*.psv"))
    df_all = []
    for f in files:
        df = pd.read_csv(f, sep='|')
        df['PatientID'] = os.path.basename(f).replace('.psv','')
        df_all.append(df)
    df_all = pd.concat(df_all, ignore_index=True)

    # Build feature snapshots for all offsets
    dfs = []
    for offset in OFFSET_RANGE:
        snap = get_sliding_windows(df_all, offset, window_size)
        if not snap.empty:
            dfs.append(snap)
    df_snap = pd.concat(dfs, ignore_index=True)

    # Split patients into train/test
    pids = df_snap['PatientID'].unique()
    train_ids, test_ids = train_test_split(pids, test_size=0.2, random_state=42)
    train_mask = df_snap['PatientID'].isin(train_ids)
    train_df = df_snap[train_mask]
    test_df = df_snap[~train_mask]

    X_train = train_df.drop(columns=['SepsisLabel','PatientID','Hour'])
    y_train = train_df['SepsisLabel']
    X_test = test_df.drop(columns=['SepsisLabel','PatientID','Hour'])
    y_test = test_df['SepsisLabel']

    # Hyperparameter tuning once
    rf = RandomForestClassifier(random_state=41)
    param_space = {
        'n_estimators': (50, 300),
        'max_depth': (5, 50),
        'min_samples_split': (2, 20),
        'min_samples_leaf': (1, 10)
    }
    search = BayesSearchCV(
        rf, param_space, n_iter=10, cv=3,
        scoring='f1', n_jobs=-1, random_state=42
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    # Predict probabilities per sample
    y_prob = best_model.predict_proba(X_test)[:,1]
    y_pred = (y_prob > threshold).astype(int)

    # Save predictions file
    out_df = test_df[['PatientID','Hour']].copy()
    out_df['Probability'] = y_prob
    out_df.to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv}")

    # Compute and print metrics
    metrics = compute_physionet_metrics(y_test.tolist(), y_pred.tolist(), y_prob.tolist(), test_df['PatientID'].tolist())
    for k,v in metrics.items():
        print(f"{k}: {v:.3f}")

if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--zip', dest='zip_path', required=True,
                   help='Path to cleaned_dataset.zip')
    p.add_argument('--window-size', dest='window_size', type=int,
                   default=WINDOW_SIZE_DEFAULT, help='Sliding window size')
    p.add_argument('--threshold', dest='threshold', type=float,
                   default=0.4, help='Decision threshold')
    p.add_argument('--out', dest='output_csv', default='predictions.csv',
                   help='Output CSV of PatientID,Hour,Probability')
    args = p.parse_args()

    run_pipeline(args.zip_path, args.window_size, args.threshold, args.output_csv)


