import pandas as pd
import xgboost as xgb
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix
from sklearn.model_selection import train_test_split
from class_balance import load_data

def create_windows(df, window_size, step_size, prediction_horizon):
    df = df.drop(columns=["index", "Gender", "Age_18-44", "Age_45-59", "Age_60-64", "Age_65-74", "Age_75-79", "Age_80-89", "HospAdmTime"])
    df = df.sort_values(["patient_id", "ICULOS"])
    X_windows = []
    y_windows = []
    patient_ids = []

    for pid, patient_data in df.groupby("patient_id"):
        values = patient_data.drop(columns=["patient_id", "ICULOS", "SepsisLabel"]).values
        sepsis_labels = patient_data["SepsisLabel"].values

        for i in range(window_size, len(patient_data) - prediction_horizon, step_size):
            window_features = values[i-window_size:i].flatten()
            future_sepsis = np.any(sepsis_labels[i:i+prediction_horizon])

            X_windows.append(window_features)
            y_windows.append(future_sepsis)
            patient_ids.append(pid)

    return np.array(X_windows), np.array(y_windows), np.array(patient_ids)

def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy:", accuracy)
    precision = precision_score(y_true, y_pred)
    print("Precision:", precision)
    aps = average_precision_score(y_true, y_pred)
    print("Average precision score:", aps)
    recall = recall_score(y_true, y_pred)
    print("Recall:", recall)
    f1 = f1_score(y_true, y_pred)
    print("F1 Score:", f1)
    auc = roc_auc_score(y_true, y_pred)
    print("AUC-ROC:", auc)
    mae = mean_absolute_error(y_true, y_pred)
    print("Mean Absolute Error:", mae)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print("Root Mean Squared Error:", rmse)

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.show()


# Load and process data
data = load_data("/home/dipl0id/Documents/cleaned_dataset")

# Create windows (6h lookback, 6h prediction horizon)
X, y, patient_ids = create_windows(data, window_size=6, step_size=1, prediction_horizon=6)

# Balance classes
positive_indices = np.where(y == 1)[0]
negative_indices = np.where(y == 0)[0]
negative_subset = np.random.choice(negative_indices, size=2*len(positive_indices), replace=False)

balanced_indices = np.concatenate([positive_indices, negative_subset])
X_balanced = X[balanced_indices]
y_balanced = y[balanced_indices]
patient_ids_balanced = patient_ids[balanced_indices]

# Split by patient IDs to avoid leakage
unique_patients = np.unique(patient_ids_balanced)
train_patients, test_patients = train_test_split(unique_patients, test_size=0.2, random_state=42)

train_mask = np.isin(patient_ids_balanced, train_patients)
test_mask = np.isin(patient_ids_balanced, test_patients)

X_train, X_test = X_balanced[train_mask], X_balanced[test_mask]
y_train, y_test = y_balanced[train_mask], y_balanced[test_mask]

# Create validation set from training patients
train_patient_ids = patient_ids_balanced[train_mask]
val_patients = np.random.choice(np.unique(train_patient_ids),
                              size=int(0.2*len(np.unique(train_patient_ids))),
                              replace=False)
val_mask = np.isin(train_patient_ids, val_patients)

X_trn, X_val = X_train[~val_mask], X_train[val_mask]
y_trn, y_val = y_train[~val_mask], y_train[val_mask]

# Calculate class weight ratio
ratio = len(y_trn[y_trn == 0]) / len(y_trn[y_trn == 1])

# Train model
param = {
    'max_depth': 5,
    'learning_rate': 0.1,
    'objective': 'binary:logistic',
    'scale_pos_weight': ratio,
    'eval_metric': 'aucpr',
    'early_stopping_rounds': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0
}

model = xgb.XGBClassifier(**param)
model.fit(X_trn, y_trn,
         eval_set=[(X_val, y_val)],
         verbose=True)

# Evaluate
y_pred_proba = model.predict_proba(X_test)[:, 1]

y_pred = (y_pred_proba >= 0.38).astype(int)
evaluate_model(y_test, y_pred)

# Feature importance
xgb.plot_importance(model)
plt.show()
