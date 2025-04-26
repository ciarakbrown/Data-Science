import os
import zipfile
import glob
import pandas as pd
import numpy as np
import tempfile
import shutil
import argparse
import sys
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, fbeta_score, make_scorer
from skopt import BayesSearchCV
from skopt.space import Integer, Real

# Add utils path for evaluate_sepsis_score
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from evaluate_sepsis_score import evaluate_sepsis_score

f2_scorer = make_scorer(fbeta_score, beta=2.5)

OFFSET_RANGE = range(-12, 0)

XGB_PARAM_SPACE = {
    'max_depth':        Integer(3, 15),
    'learning_rate':    Real(0.005, 0.3, prior='log-uniform'),
    'gamma':            Real(0, 100),
    'min_child_weight': Integer(1, 10),
    'subsample':        Real(0.5, 1.0),
    'colsample_bytree': Real(0.5, 1.0),
    'reg_alpha':        Real(0, 150),
    'reg_lambda':       Real(0, 150),
    'max_delta_step':   Integer(0, 100),
    'n_estimators':     Integer(50, 300),
}


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

def find_best_threshold(df_truth, df_preds, beta=2.5):
    merged = pd.merge(df_truth, df_preds, on=["PatientID", "Hour"])
    y_true = merged["SepsisLabel"].values
    y_prob = merged["PredictedProbability"].values

    best_thresh = 0.5
    best_score = -1
    thresholds = []
    recalls = []
    f2_scores = []
    for t in np.linspace(0, 1, 101):
        y_pred = (y_prob >= t).astype(int)
        score = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
        recall = recall_score(y_true, y_pred)
        thresholds.append(t)
        recalls.append(recall)
        f2_scores.append(score)
        if score > best_score:
            best_score = score
            best_thresh = t

    plt.figure(figsize=(10, 5))
    plt.plot(thresholds, f2_scores, label='F2 Score', color='blue')
    plt.plot(thresholds, recalls, label='Recall', color='green')
    plt.axvline(best_thresh, linestyle='--', color='red', label=f'Selected Threshold = {best_thresh:.2f}')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Threshold Tuning')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("threshold_tuning_plot.png", dpi=1200)
    plt.close()

    return best_thresh, best_score

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
        df['PatientID'] = os.path.basename(f).replace('.psv','')
        df_all.append(df)
    df_all = pd.concat(df_all, ignore_index=True)

    pids = df_all['PatientID'].unique()
    train_ids, test_ids = train_test_split(pids, test_size=0.2, random_state=42)

    best_score = -np.inf
    best_cfg   = None
    best_model = None

    for ws in window_sizes:
        snaps = []
        for off in OFFSET_RANGE:
            s = get_sliding_windows(df_all, off, ws)
            if not s.empty:
                snaps.append(s)
        df_snap = pd.concat(snaps, ignore_index=True)
        train_df = df_snap[df_snap['PatientID'].isin(train_ids)]

        X = train_df.drop(columns=['SepsisLabel','PatientID'])
        y = train_df['SepsisLabel']

        neg_count = (y == 0).sum()
        pos_count = (y == 1).sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

        xgb = XGBClassifier(use_label_encoder=False,
            eval_metric='logloss',
            random_state=41,
            scale_pos_weight=scale_pos_weight)
        bayes = BayesSearchCV(
            xgb,
            XGB_PARAM_SPACE,
            n_iter=50,
            cv=5,
            scoring=f2_scorer,
            n_jobs=-1,
            random_state=42
        )
        bayes.fit(X, y)
        print(f"WS={ws:2d} → CV F2 = {bayes.best_score_:.3f}")

        if bayes.best_score_ > best_score:
            best_score = bayes.best_score_
            best_cfg   = (ws, bayes.best_params_)
            best_model = bayes.best_estimator_

    print(f"\nSelected window_size={best_cfg[0]}, params={best_cfg[1]} (CV F2={best_score:.3f})\n")

    test_all     = df_all[df_all['PatientID'].isin(test_ids)]
    df_test_snap = get_all_windows(test_all, best_cfg[0])

    X_test = df_test_snap.drop(columns=['PatientID','Hour'])
    X_test = X_test.reindex(columns=best_model.get_booster().feature_names, fill_value=0)
    probs  = best_model.predict_proba(X_test)[:,1]

    df_preds = df_test_snap[['PatientID','Hour']].copy()
    df_preds['PredictedProbability'] = probs

    os.makedirs(output_dir, exist_ok=True)
    df_preds.to_csv(os.path.join(output_dir, "all_patient_probs.csv"), index=False)
    print(f"Saved probabilities → {output_dir}/all_patient_probs.csv")

    df_truth = test_all[['PatientID','SepsisLabel']].copy()
    df_truth['Hour'] = df_truth.groupby("PatientID").cumcount()

    best_thresh, best_f2 = find_best_threshold(df_truth, df_preds)
    print(f"\nBest threshold by F2 score: {best_thresh:.3f} (F2={best_f2:.4f})\n")

    metrics = compute_physionet_metrics_multitime(df_truth, df_preds, threshold=best_thresh)
    print("=== Evaluation Metrics ===")
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
