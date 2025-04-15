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
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Add utils path for evaluate_sepsis_score
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from evaluate_sepsis_score import evaluate_sepsis_score

# Sliding window over patient data
def get_sliding_windows(df_all, offset, window_size=6):
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
        flat_features = window.mean(numeric_only=True)
        flat_features["SepsisLabel"] = label
        flat_features["PatientID"] = patient_id
        rows.append(flat_features)

    return pd.DataFrame(rows)

# Formatting and saving of patient data ready for eval using utility function
def save_predictions_and_labels(y_true_list, y_pred_list, y_prob_list, patient_ids):
    label_dir = tempfile.mkdtemp()
    pred_dir = tempfile.mkdtemp()

    for i, pid in enumerate(patient_ids):
        df_label = pd.DataFrame({"SepsisLabel": [y_true_list[i]] * 10})
        df_pred = pd.DataFrame({
            "PredictedProbability": [y_prob_list[i]] * 10,
            "PredictedLabel": [y_pred_list[i]] * 10
        })
        df_label.to_csv(os.path.join(label_dir, f"{pid}.psv"), sep='|', index=False)
        df_pred.to_csv(os.path.join(pred_dir, f"{pid}.psv"), sep='|', index=False)

    return label_dir, pred_dir

# Evaluate RF with physionet utility function
def compute_physionet_utility(y_true_list, y_pred_list, y_prob_list, patient_ids):
    label_dir, pred_dir = save_predictions_and_labels(y_true_list, y_pred_list, y_prob_list, patient_ids)
    try:
        _, _, _, _, utility = evaluate_sepsis_score(label_dir, pred_dir)
    finally:
        shutil.rmtree(label_dir)
        shutil.rmtree(pred_dir)
    return utility

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

    df_all = []
    for path in file_paths:
        df = pd.read_csv(path, sep='|')
        df["PatientID"] = os.path.basename(path).replace(".psv", "")
        df_all.append(df)
    df_all = pd.concat(df_all, ignore_index=True)

    print(f"Loaded {len(df_all['PatientID'].unique())} patients.")

    results = []
    best_model_data = None

    for offset in range(-12, 0):
        df_snap = get_sliding_windows(df_all, offset, window_size=6)
        if df_snap.empty:
            print(f"Offset {offset:+}: No valid windows.")
            continue

        X = df_snap.drop(columns=["SepsisLabel", "PatientID"])
        y = df_snap["SepsisLabel"]

        train_ids, test_ids = train_test_split(
            df_snap["PatientID"].unique(),
            test_size=0.2,
            stratify=y,
            random_state=42
        )

        train_mask = df_snap["PatientID"].isin(train_ids)
        X_train, X_test = X[train_mask], X[~train_mask]
        y_train, y_test = y[train_mask], y[~train_mask]
        pids_test = df_snap["PatientID"][~train_mask].tolist()

        # Hyperparameter tuning 
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        rf = RandomForestClassifier(random_state=41)
        search = RandomizedSearchCV(rf, param_grid, n_iter=10, cv=3, n_jobs=-1, random_state=42)
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        y_probs = best_model.predict_proba(X_test)[:, 1]
        y_pred = (y_probs > 0.4).astype(int)

        utility = compute_physionet_utility(y_test.tolist(), y_pred.tolist(), y_probs.tolist(), pids_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"[{offset:+}h] Utility={utility:.3f} | Recall={rec:.3f} | Precision={prec:.3f} | Acc={acc:.3f} | F1={f1:.3f}")
        results.append({
            "Offset": -offset,  
            "Utility": utility,
            "Recall": rec,
            "Precision": prec,
            "Accuracy": acc,
            "F1": f1,
            "ConfMatrix": confusion_matrix(y_test, y_pred)
        })

    results_df = pd.DataFrame(results)
    plt.figure(figsize=(8, 5))
    plt.plot(results_df['Offset'], results_df['Utility'], marker='o')
    plt.title("PhysioNet Utility vs. Prediction Time Offset")
    plt.xlabel("Hours Before Diagnosis")
    plt.ylabel("Utility Score")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    best = results_df.loc[results_df['Utility'].idxmax()]
    print("\nBest Offset:")
    print(best)

    # Plot confusion matrix of best offset
    plt.figure(figsize=(5, 4))
    sns.heatmap(best['ConfMatrix'], annot=True, fmt="d", cbar=False, cmap="Blues",
                xticklabels=["No Sepsis", "Sepsis"],
                yticklabels=["No Sepsis", "Sepsis"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix (Best Offset: {best['Offset']}h Before Diagnosis)")
    plt.tight_layout()
    plt.show()

# Run code
if __name__ == "__main__":
    run_pipeline()
