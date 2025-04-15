import os
import zipfile
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

#  Utility Function from physionet
def compute_prediction_utility(labels, predictions):
    total_utility = 0
    for true_labels, pred_labels in zip(labels, predictions):
        try:
            first_correct = next(i for i, (t, p) in enumerate(zip(true_labels, pred_labels)) if t == p == 1)
            utility = 1.0 / (first_correct + 1)
        except StopIteration:
            utility = 0
        total_utility += utility
    return total_utility / len(labels)

# Patient extractor 
def get_patient_snapshots(df_all, offset):
    snapshots = []

    for patient_id, group in df_all.groupby("PatientID"):
        group = group.reset_index(drop=True)
        sepsis_indices = group.index[group["SepsisLabel"] == 1].tolist()

        if sepsis_indices:
            diagnosis_time = sepsis_indices[0]
            snapshot_time = diagnosis_time + offset
            label = 1
        else:
            if len(group) + offset < 0 or len(group) < 1:
                continue
            snapshot_time = len(group) + offset
            label = 0

        if 0 <= snapshot_time < len(group):
            snapshot = group.loc[snapshot_time].copy()
            snapshot["SepsisLabel_Offset"] = label
            snapshot["SnapshotOffset"] = offset
            snapshots.append(snapshot)

    return pd.DataFrame(snapshots)

# Train model
def evaluate_rf_at_offset(df_snapshots, offset, threshold=0.4):
    non_features = ['index', 'SepsisLabel', 'SepsisLabel_Offset', 'PatientID', 'patient_id', 'SnapshotOffset']
    feature_cols = [col for col in df_snapshots.columns if col not in non_features]

    X = df_snapshots[feature_cols].fillna(df_snapshots[feature_cols].mean())
    y = df_snapshots["SepsisLabel_Offset"]

    train_ids, test_ids = train_test_split(
        df_snapshots["PatientID"].unique(),
        test_size=0.2,
        stratify=y,
        random_state=42
    )
    train_mask = df_snapshots["PatientID"].isin(train_ids)
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]
    patient_ids_test = df_snapshots["PatientID"][~train_mask]

    clf = RandomForestClassifier(n_estimators=100, random_state=41)
    clf.fit(X_train, y_train)

    if len(clf.classes_) == 2:
        y_probs = clf.predict_proba(X_test)[:, 1]
    else:
        only_class = clf.classes_[0]
        y_probs = np.ones_like(y_test, dtype=float) if only_class == 1 else np.zeros_like(y_test, dtype=float)

    y_pred = (y_probs > threshold).astype(int)

    labels, preds = [], []
    for pid, true_y, pred_y in zip(patient_ids_test, y_test, y_pred):
        labels.append([true_y])
        preds.append([pred_y])

    utility_score = compute_prediction_utility(labels, preds)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    return {
        'Offset': offset,
        'Utility': utility_score,
        'Recall': rec,
        'Precision': prec,
        'Accuracy': acc,
        'F1': f1,
        'ConfusionMatrix': cm
    }

# Main Pipeline 
def run_utility_optimization_pipeline(zip_path):
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
    for offset in range(-12, 0):  # only early prediction offsets (before diagnosis)
        df_snap = get_patient_snapshots(df_all, offset)
        if df_snap.empty:
            print(f"Offset {offset:+}: No valid snapshots.")
            continue

        scores = evaluate_rf_at_offset(df_snap, offset)
        print(f"[{offset:+}h] Utility={scores['Utility']:.3f} | Recall={scores['Recall']:.3f} | "
              f"Precision={scores['Precision']:.3f} | Acc={scores['Accuracy']:.3f} | F1={scores['F1']:.3f}")
        results.append(scores)

    results_df = pd.DataFrame(results)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(results_df['Offset'], results_df['Utility'], marker='o')
    plt.title("Utility Score vs. Prediction Time Offset")
    plt.xlabel("Hour of Prediction (Relative to Actual Sepsis Diagnosis")
    plt.ylabel("Utility Score")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    best = max(results, key=lambda r: r['Utility'])
    print("\nBest Offset (overall):")
    print(f"Offset: {best['Offset']} hours | Utility: {best['Utility']:.3f}")

    plt.figure(figsize=(5, 4))
    sns.heatmap(best['ConfusionMatrix'], cmap = "Blues", annot=True, fmt="d", cbar=False,
                xticklabels=["No Sepsis", "Sepsis"],
                yticklabels=["No Sepsis", "Sepsis"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix for Best Prediction at {best['Offset']}h)")
    plt.tight_layout()
    plt.show()

    return results_df

# Run the code
if __name__ == "__main__":
    zip_path = input("Please enter the path to cleaned_dataset.zip: ")
    if not os.path.isfile(zip_path):
        print("Error: File not found.")
    else:
        run_utility_optimization_pipeline(zip_path)
