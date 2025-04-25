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

# Add utility path for evaluate_sepsis_score
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from evaluate_sepsis_score import evaluate_sepsis_score

OFFSET_RANGE = range(-12, 0)

def get_sliding_windows(df_all, offset, window_size):
    rows = []
    for pid, group in df_all.groupby("PatientID"):
        g = group.reset_index(drop=True)
        sepsis_idxs = g.index[g["SepsisLabel"] == 1].tolist()
        if sepsis_idxs:
            diag = sepsis_idxs[0]
            snap = diag + offset
            label = 1
        else:
            snap = len(g) + offset
            label = 0
        start, end = snap - window_size + 1, snap + 1
        if 0 <= start and end <= len(g):
            w = g.iloc[start:end]
            feats = w.mean(numeric_only=True)
            feats["SepsisLabel"] = label
            feats["PatientID"] = pid
            rows.append(feats)
    return pd.DataFrame(rows)

def get_all_windows(df_all, window_size):
    rows = []
    for pid, group in df_all.groupby("PatientID"):
        g = group.reset_index(drop=True)
        for t in range(window_size - 1, len(g)):
            w = g.iloc[t - window_size + 1 : t + 1]
            feats = w.mean(numeric_only=True)
            feats["PatientID"] = pid
            feats["Hour"] = t
            rows.append(feats)
    return pd.DataFrame(rows)

def compute_physionet_metrics_multitime(df_truth, df_preds, threshold=0.5):
    df = pd.merge(df_truth, df_preds, on=["PatientID", "Hour"])
    df["PredictedLabel"] = (df["PredictedProbability"] > threshold).astype(int)

    label_dir = tempfile.mkdtemp()
    pred_dir = tempfile.mkdtemp()

    try:
        for pid, grp in df.groupby("PatientID"):
            grp = grp.sort_values("Hour")
            grp[["SepsisLabel"]].to_csv(os.path.join(label_dir, f"{pid}.psv"), sep="|", index=False)
            grp[["PredictedProbability", "PredictedLabel"]].to_csv(os.path.join(pred_dir, f"{pid}.psv"), sep="|", index=False)
        auroc, auprc, accuracy, f1, utility = evaluate_sepsis_score(label_dir, pred_dir)
        recall = recall_score(df["SepsisLabel"], df["PredictedLabel"], zero_division=0)
    finally:
        shutil.rmtree(label_dir)
        shutil.rmtree(pred_dir)

    return {
        'Utility': utility,
        'AUROC': auroc,
        'AUPRC': auprc,
        'Accuracy': accuracy,
        'Recall': recall,
        'F1': f1
    }

def run_pipeline(zip_path, window_sizes, threshold, output_dir):
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"{zip_path} not found")
    extract_dir = "./sepsis_dataset"
    if not os.path.exists(extract_dir):
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_dir)

    data_dir = os.path.join(extract_dir, "cleaned_dataset")
    files = glob.glob(os.path.join(data_dir, "*.psv"))
    df_all = []
    for f in files:
        df = pd.read_csv(f, sep='|')
        df["PatientID"] = os.path.basename(f).replace(".psv", "")
        df_all.append(df)
    df_all = pd.concat(df_all, ignore_index=True)

    pids = df_all["PatientID"].unique()
    train_ids, test_ids = train_test_split(pids, test_size=0.2, random_state=42)

    best_score = -np.inf
    best_cfg = None
    best_model = None

    for ws in window_sizes:
        snaps = []
        for offset in OFFSET_RANGE:
            s = get_sliding_windows(df_all, offset, ws)
            if not s.empty:
                snaps.append(s)
        df_snap = pd.concat(snaps, ignore_index=True)
        train_df = df_snap[df_snap["PatientID"].isin(train_ids)]

        X = train_df.drop(columns=["SepsisLabel", "PatientID"])
        y = train_df["SepsisLabel"]

        rf = RandomForestClassifier(random_state=41)
        bayes = BayesSearchCV(
            rf,
            {
                "n_estimators": (50, 300),
                "max_depth": (5, 50),
                "min_samples_split": (2, 20),
                "min_samples_leaf": (1, 10)
            },
            n_iter=10, cv=3, scoring="f1", n_jobs=-1, random_state=42
        )
        bayes.fit(X, y)
        print(f"WS={ws} → CV F1 = {bayes.best_score_:.3f}")
        if bayes.best_score_ > best_score:
            best_score = bayes.best_score_
            best_cfg = (ws, bayes.best_params_)
            best_model = bayes.best_estimator_

    print(f"\nSelected window_size={best_cfg[0]}, params={best_cfg[1]} (CV F1={best_score:.3f})\n")

    test_all = df_all[df_all["PatientID"].isin(test_ids)]
    df_test_snap = get_all_windows(test_all, best_cfg[0])

    X_test = df_test_snap.drop(columns=["PatientID", "Hour"])
    X_test = X_test.reindex(columns=best_model.feature_names_in_, fill_value=0)
    probs = best_model.predict_proba(X_test)[:, 1]

    df_preds = df_test_snap[["PatientID", "Hour"]].copy()
    df_preds["PredictedProbability"] = probs

    os.makedirs(output_dir, exist_ok=True)
    df_preds.to_csv(os.path.join(output_dir, "all_patient_probs.csv"), index=False)
    print(f"Saved predictions to {output_dir}/all_patient_probs.csv")

    df_truth = test_all[["PatientID", "SepsisLabel"]].copy()
    df_truth["Hour"] = df_truth.groupby("PatientID").cumcount()

    metrics = compute_physionet_metrics_multitime(df_truth, df_preds, threshold=threshold)

    print("\nEvaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k:10s}: {v:.4f}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--zip', dest='zip_path', required=True, help='Path to cleaned_dataset.zip')
    p.add_argument('--window-sizes', dest='window_sizes', nargs='+', type=int, default=[6, 12, 24], help='Candidate window sizes')
    p.add_argument('--threshold', dest='threshold', type=float, default=0.5, help='Probability threshold')
    p.add_argument('--out-dir', dest='output_dir', default='predictions', help='Directory for output CSVs')
    args = p.parse_args()

    run_pipeline(args.zip_path, args.window_sizes, args.threshold, args.output_dir)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--zip',          dest='zip_path',     required=True,
                   help='Path to cleaned_dataset.zip')
    p.add_argument('--window-sizes', dest='window_sizes', nargs='+', type=int,
                   default=[6, 12, 24], help='Candidate sliding window sizes')
    p.add_argument('--threshold',    dest='threshold',    type=float,
                   default=0.5, help='Probability threshold')
    p.add_argument('--out-dir',      dest='output_dir',   default='predictions',
                   help='Directory for output CSVs')
    args = p.parse_args()

    run_pipeline(args.zip_path, args.window_sizes, args.threshold, args.output_dir)

