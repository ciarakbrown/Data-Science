import pandas as pd
import xgboost as xgb
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix,
    precision_recall_curve, roc_curve
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from class_balance import load_data
from skopt import BayesSearchCV
from skopt.space import Real, Integer

def create_windows(df, window_size, step_size, prediction_horizon):
    drop_cols = ["HospAdmTime", "SBP", "DBP", "FiO2", "SI"]
    df = df.drop(columns=drop_cols).sort_values(["patient_id", "ICULOS"])

    static_cols = ["Gender", "Age"]
    time_varying_cols = [col for col in df.columns
                         if col not in ["patient_id", "ICULOS", "SepsisLabel"] + static_cols]

    X_windows = []
    y_windows = []
    patient_ids = []
    feature_names = []

    for col in time_varying_cols:
        feature_names.extend([
            f"{col}_mean",
            f"{col}_std",
            f"{col}_latest",
            f"{col}_trend"
        ])
    feature_names.extend(static_cols)

    for pid, patient_data in df.groupby("patient_id"):
        static_features = patient_data[static_cols].iloc[0].values
        time_varying_data = patient_data[time_varying_cols].values
        sepsis_labels = patient_data["SepsisLabel"].values

        for window_end in range(window_size, len(patient_data) - prediction_horizon, step_size):
            window_start = window_end - window_size
            window = time_varying_data[window_start:window_end, :]

            features = []
            for col_idx in range(window.shape[1]):
                col = window[:, col_idx]
                features.extend([
                    np.mean(col),
                    np.std(col),
                    col[-1],
                    col[-1] - col[0]
                ])
            features.extend(static_features)

            label_start = window_end
            label_end = window_end + prediction_horizon
            future_sepsis = np.any(sepsis_labels[label_start:label_end])

            X_windows.append(features)
            y_windows.append(future_sepsis)
            patient_ids.append(pid)

    return (
        np.array(X_windows, dtype=np.float32),
        np.array(y_windows),
        np.array(patient_ids),
        feature_names
    )

def evaluate_model(y_true, y_pred_proba, threshold=None, plot_pr=True):
    metrics = {
        'auc_roc': roc_auc_score(y_true, y_pred_proba),
        'average_precision': average_precision_score(y_true, y_pred_proba)
    }

    desired_recall = 0.85
    if threshold is None:
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
        thresholds = np.append(thresholds, 1)

        valid_indices = np.where(recalls[:-1] >= desired_recall)[0]
        if valid_indices.size > 0:
            optimal_idx = valid_indices[np.argmax(precisions[valid_indices])]
            threshold = thresholds[optimal_idx]
        else:
            # Fallback: use ROC curve and maximize Youden's J statistic
            fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_proba)
            youden_j = tpr - fpr
            optimal_idx = np.argmax(youden_j)
            threshold = roc_thresholds[optimal_idx]

    y_pred = (y_pred_proba >= threshold).astype(int)
    metrics.update({
        'threshold': threshold,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    })

    return metrics

# Load data
data = load_data("/home/dipl0id/Documents/clean_out_sofa")
X, y, patient_ids, feature_names = create_windows(data, window_size=6, step_size=1, prediction_horizon=6)

# Balance classes
positive_indices = np.where(y == 1)[0]
negative_indices = np.where(y == 0)[0]
negative_subset = np.random.choice(negative_indices, size=2*len(positive_indices), replace=False)

balanced_indices = np.concatenate([positive_indices, negative_subset])
X_balanced = X[balanced_indices]
y_balanced = y[balanced_indices]
pid_balanced = patient_ids[balanced_indices]

# Get per-patient labels (label is 1 if patient ever had a positive window)
patient_labels = {}
for pid in np.unique(pid_balanced):
    patient_labels[pid] = int(np.any(y_balanced[pid_balanced == pid]))

# Train/test split by patient
unique_patients = np.array(list(patient_labels.keys()))
patient_label_array = np.array([patient_labels[pid] for pid in unique_patients])

train_patients, test_patients = train_test_split(
    unique_patients, test_size=0.2, stratify=patient_label_array, random_state=42
)

train_mask = np.isin(pid_balanced, train_patients)
test_mask = np.isin(pid_balanced, test_patients)

X_train, X_test = X_balanced[train_mask], X_balanced[test_mask]
y_train, y_test = y_balanced[train_mask], y_balanced[test_mask]
pid_train = pid_balanced[train_mask]

# Get patient-level labels for stratified folds
train_unique_pids = np.unique(pid_train)
train_labels_per_patient = np.array([
    int(np.any(y_train[pid_train == pid])) for pid in train_unique_pids
])

# Calculate class ratio
ratio = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

# Define hyperparameter search space
search_spaces = {
    'max_depth': Integer(3, 15),
    'learning_rate': Real(0.005, 0.3, 'log-uniform'),
    'gamma': Real(0, 100),
    'min_child_weight': Integer(1, 10),
    'subsample': Real(0.5, 1.0),
    'colsample_bytree': Real(0.5, 1.0),
    'reg_alpha': Real(0, 150),
    'reg_lambda': Real(0, 150),
    'max_delta_step': Integer(0, 100),
    'scale_pos_weight': Real(ratio * 0.3, ratio * 2.0)
}

# Patient-level StratifiedKFold generator
cv_folds = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in skf.split(train_unique_pids, train_labels_per_patient):
    train_pids = train_unique_pids[train_idx]
    val_pids = train_unique_pids[val_idx]

    train_mask_fold = np.isin(pid_train, train_pids)
    val_mask_fold = np.isin(pid_train, val_pids)

    cv_folds.append((np.where(train_mask_fold)[0], np.where(val_mask_fold)[0]))

from sklearn.metrics import make_scorer, fbeta_score

f2_scorer = make_scorer(fbeta_score, beta=2)


class XGBoostWithEarlyStop(xgb.XGBClassifier):
    def fit(self, X, y, eval_set=None, **kwargs):
        return super().fit(X, y, eval_set=eval_set, **kwargs)

# Run Bayesian optimization with custom folds
opt = BayesSearchCV(
    estimator=XGBoostWithEarlyStop(
        objective='binary:logistic',
        eval_metric='aucpr',
        early_stopping_rounds=20,
        n_estimators=200,
        random_state=42
    ),
    search_spaces=search_spaces,
    scoring=f2_scorer,
    cv=cv_folds,
    n_iter=50,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

opt.fit(X_train, y_train, eval_set=[(X_train[val_idx], y_train[val_idx]) for _, val_idx in cv_folds], verbose=False)

optimal_thresholds = []



for fold_idx, (_, val_idx) in enumerate(cv_folds):
    y_val_true = y_train[val_idx]
    y_val_proba = opt.best_estimator_.predict_proba(X_train[val_idx])[:, 1]

    thresholds = np.linspace(0.05, 0.95, 100)
    f2_scores = []

    for t in thresholds:
        y_pred = (y_val_proba >= t).astype(int)
        f2 = fbeta_score(y_val_true, y_pred, beta=2.5)
        f2_scores.append(f2)

    best_idx = np.argmax(f2_scores)
    threshold = thresholds[best_idx]
    optimal_thresholds.append(threshold)

final_threshold = np.median(optimal_thresholds)

best_params = opt.best_params_
print("Best parameters found:", best_params)

# Train final model
final_model = xgb.XGBClassifier(
    **best_params,
    objective='binary:logistic',
    eval_metric='aucpr',
    early_stopping_rounds=20,
    random_state=42
)
final_model.fit(X_train, y_train, eval_set=[(X_train[val_idx], y_train[val_idx]) for _, val_idx in cv_folds])

# Evaluate on test set
y_test_proba = final_model.predict_proba(X_test)[:, 1]
test_metrics = evaluate_model(y_test, y_test_proba, final_threshold)

# Confusion matrix
sns.heatmap(test_metrics['confusion_matrix'],
            annot=True, fmt='d',
            xticklabels=['No Sepsis', 'Sepsis'],
            yticklabels=['No Sepsis', 'Sepsis'],
            cmap="Blues")
plt.ylabel('True')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.savefig("figures/confusion_matrix.png", dpi=1200)
plt.show()

# Test metrics
print("\nTest Set Performance:")
for k, v in test_metrics.items():
    if k != 'confusion_matrix':
        print(f"{k:>18}: {v:.4f}")

# Plot threshold metrics
thresholds = np.linspace(0.1, 0.95, 50)
recalls = [recall_score(y_test, (y_test_proba >= t).astype(int)) for t in thresholds]
bas = [fbeta_score(y_test, (y_test_proba >= t).astype(int), beta=1.5) for t in thresholds]

plt.figure(figsize=(10, 6))
plt.plot(thresholds, recalls, label='Recall')
plt.plot(thresholds, bas, label='F2 Score')
plt.axvline(x=final_threshold, linestyle="--", label="Optimal Threshold")
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Threshold Tuning on Validation Folds')
plt.legend()
plt.grid(True)
plt.savefig("figures/recall_precision.png", dpi=1200)
plt.show()

# Feature importance
xgb.plot_importance(final_model)
plt.title("Feature Importance")
plt.show()


y_test_pred = (y_test_proba >= final_threshold).astype(int)

# Get test set patient IDs (per row in test set)
pid_test = pid_balanced[test_mask]

# Get ICULOS values for each test row (need from original dataframe)
test_df = data[data['patient_id'].isin(test_patients)]
test_df = test_df.sort_values(["patient_id", "ICULOS"])
iculos_lookup = test_df.groupby("patient_id")["ICULOS"].apply(list).to_dict()

for pid in np.unique(pid_test):
    idxs = np.where(pid_test == pid)[0]

    labels_bin = y_test[idxs][::-1]         # Reverse
    preds_prob = y_test_proba[idxs][::-1]   # Reverse
    preds_bin = y_test_pred[idxs][::-1]     # Reverse
    iculos = iculos_lookup[pid][:len(labels_bin)]  # ICULOS stays in order

    # Write labels file
    with open(f"labels/{pid}.psv", "w") as f:
        f.write("SepsisLabel|ICULOS\n")
        for label, icu in zip(labels_bin, iculos):
            f.write(f"{int(label)}|{int(icu)}\n")

    # Write predictions file
    with open(f"predictions/{pid}.psv", "w") as f:
        f.write("PredictedProbability|PredictedLabel\n")
        for prob, pred in zip(preds_prob, preds_bin):
            f.write(f"{prob:.4f}|{pred}\n")
