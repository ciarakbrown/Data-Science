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

# Sliding window feature extractor for time series
WINDOW_SIZE = 6

# Prepare features at each hour for each patient
def prepare_features(df_patient, window_size=6):
    features = []
    labels = []
    for current_time in range(window_size - 1, len(df_patient)):
        window = df_patient.iloc[current_time - window_size + 1:current_time + 1]
        flat = window.mean(numeric_only=True)
        label = df_patient.loc[current_time, "SepsisLabel"]
        features.append(flat)
        labels.append(label)
    return np.array(features), np.array(labels)

# Save predictions and labels properly for PhysioNet eval
def save_predictions_and_labels(full_labels, full_preds, full_probs, patient_ids, timesteps):
    label_dir = tempfile.mkdtemp()
    pred_dir = tempfile.mkdtemp()

    idx = 0
    for pid, length in zip(patient_ids, timesteps):
        df_label = pd.DataFrame({"SepsisLabel": full_labels[idx:idx+length]})
        df_pred = pd.DataFrame({
            "PredictedProbability": full_probs[idx:idx+length],
            "PredictedLabel": full_preds[idx:idx+length]
        })
        df_label.to_csv(os.path.join(label_dir, f"{pid}.psv"), sep='|', index=False)
        df_pred.to_csv(os.path.join(pred_dir, f"{pid}.psv"), sep='|', index=False)
        idx += length

    return label_dir, pred_dir

# Evaluate properly
def compute_physionet_utility(full_labels, full_preds, full_probs, patient_ids, timesteps):
    label_dir, pred_dir = save_predictions_and_labels(full_labels, full_preds, full_probs, patient_ids, timesteps)
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

    patient_data = {}
    for path in file_paths:
        df = pd.read_csv(path, sep='|')
        pid = os.path.basename(path).replace(".psv", "")
        df["PatientID"] = pid
        patient_data[pid] = df

    print(f"Loaded {len(patient_data)} patients.")

    # Create full dataset
    X_all = []
    y_all = []
    patient_ids = []
    time_lengths = []

    for pid, df in patient_data.items():
        if len(df) >= WINDOW_SIZE:
            X_patient, y_patient = prepare_features(df, window_size=WINDOW_SIZE)
            X_all.append(X_patient)
            y_all.append(y_patient)
            patient_ids.extend([pid]*len(y_patient))
            time_lengths.append(len(y_patient))

    X_all = np.vstack(X_all)
    y_all = np.hstack(y_all)

    # Train-test split patient wise
    unique_ids = np.unique(patient_ids)
    train_ids, test_ids = train_test_split(unique_ids, test_size=0.2, random_state=42)

    train_mask = np.array([pid in train_ids for pid in patient_ids])

    X_train, X_test = X_all[train_mask], X_all[~train_mask]
    y_train, y_test = y_all[train_mask], y_all[~train_mask]
    pids_test = np.array(patient_ids)[~train_mask]

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

    # Evaluate
    utility = compute_physionet_utility(y_test.tolist(), y_pred.tolist(), y_probs.tolist(), pids_test.tolist(), [np.sum(pids_test==pid) for pid in np.unique(pids_test)])

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"Final Utility Score: {utility:.3f}")
    print(f"Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1 Score: {f1:.3f}")

    # Confusion Matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cbar=False, cmap="Blues",
                xticklabels=["No Sepsis", "Sepsis"],
                yticklabels=["No Sepsis", "Sepsis"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

# Run
if __name__ == "__main__":
    run_pipeline()

