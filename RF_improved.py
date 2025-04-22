import os
import zipfile
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import shutil
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from skopt import BayesSearchCV

# Add utils path for evaluate_sepsis_score
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from evaluate_sepsis_score import evaluate_sepsis_score

# Function to extract sliding-window snapshots
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
            # For non-sepsis patients, use last timestamp + offset
            if len(group) + offset < 0 or len(group) < window_size:
                continue
            snapshot_time = len(group) + offset
            label = 0

        start = snapshot_time - window_size + 1
        end = snapshot_time + 1

        if start < 0 or end > len(group):
            continue

        window = group.iloc[start:end].copy()
        flat_features = window.mean(numeric_only=True)
        flat_features["SepsisLabel"] = label
        flat_features["PatientID"] = patient_id
        rows.append(flat_features)

    return pd.DataFrame(rows)

# Save predictions and labels for PhysioNet evaluation
def save_predictions_and_labels(y_true_list, y_pred_list, y_prob_list, patient_ids):
    label_dir = tempfile.mkdtemp()
    pred_dir = tempfile.mkdtemp()

    for i, pid in enumerate(patient_ids):
        # replicate 10 timesteps per patient as in second script
        df_label = pd.DataFrame({"SepsisLabel": [y_true_list[i]] * 10})
        df_pred = pd.DataFrame({
            "PredictedProbability": [y_prob_list[i]] * 10,
            "PredictedLabel": [y_pred_list[i]] * 10
        })
        df_label.to_csv(os.path.join(label_dir, f"{pid}.psv"), sep='|', index=False)
        df_pred.to_csv(os.path.join(pred_dir, f"{pid}.psv"), sep='|', index=False)

    return label_dir, pred_dir

# Compute utility (and other metrics) via evaluate_sepsis_score
def compute_physionet_utility(y_true_list, y_pred_list, y_prob_list, patient_ids):
    label_dir, pred_dir = save_predictions_and_labels(y_true_list, y_pred_list, y_prob_list, patient_ids)
    try:
        auroc, auprc, accuracy, f_measure, utility = evaluate_sepsis_score(label_dir, pred_dir)
    finally:
        shutil.rmtree(label_dir)
        shutil.rmtree(pred_dir)
    return auroc, auprc, accuracy, f_measure, utility

# Main Pipeline
def run_pipeline():
    zip_path = input("Please enter the full path to the cleaned_dataset.zip file: ").strip()

    if not os.path.exists(zip_path):
        print("Invalid path to cleaned_dataset.zip. Please check the location.")
        return

    extract_dir = "./sepsis_dataset"
    if not os.path.exists(extract_dir):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

    data_dir = os.path.join(extract_dir, "cleaned_dataset")
    file_paths = glob.glob(os.path.join(data_dir, "*.psv"))

    # Load all patient data into a single DataFrame
    df_all = []
    for path in file_paths:
        df = pd.read_csv(path, sep='|')
        df['PatientID'] = os.path.basename(path).replace('.psv', '')
        df_all.append(df)
    df_all = pd.concat(df_all, ignore_index=True)

    print(f"Loaded {df_all['PatientID'].nunique()} patients.")

    results = []
    offsets = range(-12, 0)
    window_sizes = [3, 6, 9, 12]

    for offset in offsets:
        best_result_for_offset = None

        for window_size in window_sizes:
            df_snap = get_sliding_windows(df_all, offset, window_size)
            if df_snap.empty:
                continue

            X = df_snap.drop(columns=["SepsisLabel", "PatientID"])
            y = df_snap['SepsisLabel']
            groups = df_snap['PatientID']

            cv_outer = GroupKFold(n_splits=5)

            y_true_all, y_pred_all, y_prob_all, pid_all = [], [], [], []

            for train_idx, test_idx in cv_outer.split(X, y, groups):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                pid_test = groups.iloc[test_idx]

                rf = RandomForestClassifier(random_state=41)

                # Bayesian optimization search space
                param_space = {
                    'n_estimators': (50, 300),
                    'max_depth': (5, 50),
                    'min_samples_split': (2, 20),
                    'min_samples_leaf': (1, 10)
                }

                search = BayesSearchCV(
                    rf,
                    param_space,
                    n_iter=20,
                    cv=3,
                    scoring='f1',
                    n_jobs=-1,
                    random_state=42
                )
                search.fit(X_train, y_train)

                best_model = search.best_estimator_
                y_probs_fold = best_model.predict_proba(X_test)[:, 1]
                y_pred_fold = (y_probs_fold > 0.4).astype(int)

                y_true_all.extend(y_test.tolist())
                y_pred_all.extend(y_pred_fold.tolist())
                y_prob_all.extend(y_probs_fold.tolist())
                pid_all.extend(pid_test.tolist())

            # Compute metrics
            auroc, auprc, accuracy, f1, utility = compute_physionet_utility(
                y_true_all, y_pred_all, y_prob_all, pid_all
            )

            conf_mat = confusion_matrix(y_true_all, y_pred_all)

            result = {
                'Offset': -offset,
                'WindowSize': window_size,
                'Utility': utility,
                'AUROC': auroc,
                'AUPRC': auprc,
                'Accuracy': accuracy,
                'F1': f1,
                'ConfMatrix': conf_mat
            }

            # Select best window size for this offset
            if (best_result_for_offset is None) or (utility > best_result_for_offset['Utility']):
                best_result_for_offset = result

        # If a best result found, record and print
        if best_result_for_offset:
            results.append(best_result_for_offset)
            print(f"Offset: {best_result_for_offset['Offset']}h, Best Window Size: {best_result_for_offset['WindowSize']}h, Utility: {best_result_for_offset['Utility']:.3f}")

    # Summarize and plot
    results_df = pd.DataFrame(results)
    plt.figure(figsize=(8, 5))
    results_df_sorted = results_df.sort_values(by='Offset')
    plt.plot(results_df_sorted['Offset'], results_df_sorted['Utility'], marker='o')
    plt.title("PhysioNet Utility vs. Prediction Time Offset", fontsize=16)
    plt.xlabel("Hours Before Diagnosis", fontsize=14)
    plt.ylabel("Utility Score", fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run_pipeline()
