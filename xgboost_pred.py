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
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from functools import partial
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve



def create_windows(df, window_size, step_size, prediction_horizon):
    drop_cols = ["HospAdmTime"]
    df = df.drop(columns=drop_cols).sort_values(["patient_id", "ICULOS"])
    static_cols = ["Gender", "Age"]
    n_lags = 2

    X_windows = []
    y_windows = []
    patient_ids = []

    for pid, patient_data in df.groupby("patient_id"):
            static_features = patient_data[static_cols].iloc[0].values
            time_varying_data = patient_data.drop(columns=["patient_id", "ICULOS", "SepsisLabel"] + static_cols)
            values = time_varying_data.values
            sepsis_labels = patient_data["SepsisLabel"].values

            # Pre-compute rolling statistics for all columns
            rolling_stats = {}
            for col_idx in range(values.shape[1]):
                col_data = values[:, col_idx]
                rolling_stats[col_idx] = {
                    'means': [np.mean(col_data[i-window_size:i]) if i >= window_size else None
                              for i in range(len(col_data))],
                    'stds': [np.std(col_data[i-window_size:i]) if i >= window_size else None
                             for i in range(len(col_data))]
                }

            # Create windows
            for i in range(window_size + n_lags, len(patient_data) - prediction_horizon, step_size):
                window = values[i - window_size:i, :]
                features = []

                for col_idx in range(window.shape[1]):
                    col = window[:, col_idx]

                    # Base features (fixed length)
                    base_features = [
                        np.mean(col), np.std(col),
                        np.polyfit(np.arange(len(col)), col, 1)[0],  # slope
                        col[-1] - col[0]  # delta
                    ]

                    # Rolling features (fixed length = 3*n_lags)
                    rolling_feats = []
                    for lag in range(1, n_lags + 1):
                        if i - lag >= window_size:
                            prev_mean = rolling_stats[col_idx]['means'][i - lag]
                            prev_std = rolling_stats[col_idx]['stds'][i - lag]
                            rolling_feats.extend([
                                np.mean(col) - prev_mean if prev_mean is not None else 0,
                                np.std(col) - prev_std if prev_std is not None else 0,
                                col[-1] - values[i - lag, col_idx]  # pointwise delta
                            ])
                        else:
                            rolling_feats.extend([0, 0, 0])  # padding

                    features.extend(base_features + rolling_feats)

                # Add static features
                features.extend(static_features)

                # Label
                future_sepsis = np.any(sepsis_labels[i:i + prediction_horizon])

                X_windows.append(features)
                y_windows.append(future_sepsis)
                patient_ids.append(pid)

        # Convert to arrays and validate consistent shape
    X_array = np.array(X_windows, dtype=np.float32)
    y_array = np.array(y_windows)
    pid_array = np.array(patient_ids)

    return X_array, y_array, pid_array

# Custom scorer that considers both recall and F1 with more weight on recall
def recall_f1_scorer(y_true, y_pred_proba, threshold):
    y_pred = (y_pred_proba >= threshold).astype(int)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    # Weighted combination (70% recall, 30% F1)
    return 0.5 * rec + 0.5 * f1

def evaluate_with_threshold(model, X, y, threshold):
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    return {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
        'roc_auc': roc_auc_score(y, y_pred_proba),
        'average_precision': average_precision_score(y, y_pred_proba)
    }

def evaluate_model(y_true, y_pred_proba, threshold=None, plot_pr=True):
    """
    Properly evaluates classification performance using probabilities
    and optionally finds optimal threshold via precision-recall analysis.

    Args:
        y_true: True labels (array-like)
        y_pred_proba: Predicted probabilities (array-like)
        threshold: Decision threshold (if None, finds optimal via PR curve)
        plot_pr: Whether to plot precision-recall curve

    Returns:
        Dictionary of metrics and optimal threshold
    """
    # Calculate threshold-agnostic metrics first
    metrics = {
        'auc_roc': roc_auc_score(y_true, y_pred_proba),
        'average_precision': average_precision_score(y_true, y_pred_proba)
    }

    # Find optimal threshold via precision-recall if not specified
    if threshold is None:
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
        optimal_idx = np.argmax(f1_scores)
        threshold = thresholds[optimal_idx]

        if plot_pr:
            plt.figure(figsize=(8, 6))
            plt.plot(recalls, precisions, label='PR Curve')
            plt.scatter(recalls[optimal_idx], precisions[optimal_idx],
                       color='red', label=f'Optimal (F1={f1_scores[optimal_idx]:.2f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend()
            plt.show()

    # Calculate threshold-dependent metrics
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


# Load and process data
data = load_data("/home/dipl0id/Documents/clean_out")

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

search_spaces = {
    'max_depth': Integer(3, 15),
    'learning_rate': Real(0.005, 0.3, 'log-uniform'),
    'gamma': Real(0, 5),
    'min_child_weight': Integer(1, 10),
    'subsample': Real(0.5, 1.0),
    'colsample_bytree': Real(0.5, 1.0),
    'reg_alpha': Real(0, 10),
    'reg_lambda': Real(0, 10),
    'max_delta_step': Integer(0, 10),
    'scale_pos_weight': Real(ratio*0.5, ratio*1.5)  # Search around the calculated ratio
}

def custom_scorer(estimator, X, y):
    y_pred_proba = estimator.predict_proba(X)[:, 1]

    # Find optimal threshold that maximizes our custom metric
    thresholds = np.linspace(0.1, 0.5, 20)
    scores = [recall_f1_scorer(y, y_pred_proba, t) for t in thresholds]
    best_threshold = thresholds[np.argmax(scores)]

    return np.max(scores)

class XGBoostWithEarlyStop(xgb.XGBClassifier):
    def fit(self, X, y, eval_set=None, **kwargs):
        return super().fit(X, y, eval_set=eval_set, **kwargs)


from sklearn.model_selection import KFold
opt = BayesSearchCV(
    estimator=XGBoostWithEarlyStop(
        objective='binary:logistic',
        eval_metric='aucpr',
        early_stopping_rounds=20,
        n_estimators=200,
        random_state=42
    ),
    search_spaces=search_spaces,
    scoring="average_precision",
    cv=3,
    n_iter=50,  # Number of Bayesian optimization iterations
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# Fit the model
opt.fit(X_trn, y_trn, eval_set=[(X_val, y_val)], verbose=False)

# Get the best parameters
best_params = opt.best_params_
print("Best parameters found:", best_params)

# Train final model with best parameters
final_model = xgb.XGBClassifier(
    **best_params,
    objective='binary:logistic',
    eval_metric='aucpr',
    early_stopping_rounds=20,
    random_state=42,

)
final_model.fit(X_trn, y_trn, eval_set=[(X_val, y_val)], verbose=True)

# Find optimal threshold on validation set
y_val_proba = final_model.predict_proba(X_val)[:, 1]
thresholds = np.linspace(0.1, 0.5, 50)
scores = [recall_f1_scorer(y_val, y_val_proba, t) for t in thresholds]
precisions, recalls, pr_thresholds = precision_recall_curve(y_val, y_val_proba)
f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)



# Evaluate on test set with optimal threshold
test_metrics = evaluate_model(
    y_test,
    final_model.predict_proba(X_test)[:, 1],  # Use probabilities!
    threshold=None  # Auto-find optimal threshold
)
sns.heatmap(test_metrics['confusion_matrix'],
            annot=True, fmt='d',
            xticklabels=['No Sepsis', 'Sepsis'],
            yticklabels=['No Sepsis', 'Sepsis'])
plt.ylabel('True')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
print("Test Set Performance:")
for k, v in test_metrics.items():
    if k != 'confusion_matrix':
        print(f"{k:>18}: {v:.4f}")
# Plot threshold vs metrics
recalls = [recall_score(y_val, (y_val_proba >= t).astype(int)) for t in thresholds]
f1s = [f1_score(y_val, (y_val_proba >= t).astype(int)) for t in thresholds]

plt.figure(figsize=(10, 6))
plt.plot(thresholds, recalls, label='Recall')
plt.plot(thresholds, f1s, label='F1 Score')
plt.ylabel('Score')
plt.title('Threshold Selection')
plt.legend()
plt.show()

# Feature importance
xgb.plot_importance(final_model)
plt.show()
