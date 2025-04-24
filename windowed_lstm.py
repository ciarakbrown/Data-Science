import numpy as np
import pandas as pd
import os
import glob
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, roc_auc_score, ConfusionMatrixDisplay
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
import tempfile

np.random.seed(42)
tf.random.set_seed(42)

def load_data(path):
    """
    Loads patient data from PSV files and returns the full dataset with unique patient IDs.
    This function ensures that if multiple files share the same filename (basename),
    only the first encountered file is loaded.
    """
    # Get all PSV file paths
    files = glob.glob(os.path.join(path, '*.psv'))

    # Create a dictionary to hold the first occurrence for each basename.
    unique_files = {}
    for file in files:
        base_name = os.path.basename(file)
        if base_name not in unique_files:
            unique_files[base_name] = file

    # Optionally, print out the number of unique files detected
    print("Unique patient files found:", len(unique_files))

    # Now load only the unique files.
    data_frames = []
    for base_name, file in unique_files.items():
        df = pd.read_csv(file, sep='|')
        df['patient_id'] = base_name  # Use the filename as patient_id
        data_frames.append(df)

    full_dataset = pd.concat(data_frames, ignore_index=True)
    full_dataset = full_dataset.drop(columns=['HospAdmTime', 'index'], errors='ignore')
    return full_dataset

path = "/content/drive/MyDrive/cleaned_dataset"
df = load_data(path)
df = df.drop(columns=['SBP', 'DBP'], errors='ignore')
# df = df.drop(columns=['Age_18-44', 'Age_45-59', 'Age_60-64', 'Age_65-74', 'Age_75-79', 'Age_80-89', 'Gender'], errors='ignore')
print(df.head())

WINDOW_SIZE = 18
HORIZON = 1

def extract_windows(data, labels, window_size, horizon=1):
    """
    Build sliding windows of length `window_size` over `data`
    and pull the label `horizon` steps after each window end.
    """
    T = data.shape[0]
    n_windows = T - window_size - (horizon - 1)
    if n_windows <= 0:
        return np.empty((0, window_size, data.shape[1])), np.empty((0,))

    wins = []
    labs = []
    for start in range(n_windows):
        end = start + window_size
        wins.append(data[start:end])
        labs.append(labels[end + horizon - 1])
    return np.stack(wins), np.array(labs)


def sliding_windows(
    df,
    patient_id_col='patient_id',
    label_col='sepsis',
    feature_cols=None,
    window_size=WINDOW_SIZE,
    horizon=HORIZON
):
    """
    For each patient, slide a fixed-length window over their entire record.
    Returns all windows, their labels, and corresponding patient IDs.
    """
    if feature_cols is None:
        exclude = {patient_id_col, label_col}
        feature_cols = [c for c in df.columns if c not in exclude]

    all_wins = []
    all_labs = []
    all_pids = []

    for pid, grp in df.groupby(patient_id_col):
        grp = grp.sort_index()  # assume index reflects time ordering

        data_array  = grp[feature_cols].to_numpy()
        label_array = grp[label_col].to_numpy().astype(int)

        # slide windows over the full patient record
        wins, labs = extract_windows(
            data_array,
            label_array,
            window_size=window_size,
            horizon=horizon
        )
        if wins.shape[0] == 0:
            continue

        all_wins.append(wins)
        all_labs.append(labs)
        all_pids.extend([pid] * len(labs))

    if all_wins:
        windows    = np.concatenate(all_wins, axis=0)
        labels     = np.concatenate(all_labs, axis=0)
    else:
        windows = np.empty((0, window_size, len(feature_cols)))
        labels  = np.empty((0,))

    return windows, labels, all_pids


windows, labels, patient_ids = sliding_windows(
    df,
    patient_id_col='patient_id',
    label_col='SepsisLabel'
)
print(windows.shape, labels.shape, len(patient_ids))

# Make train/test splits
def make_train_test_splits(windows, labels, patient_ids, test_split=0.2):

  split_size = int(len(windows) * (1-test_split))
  train_windows = windows[:split_size]
  train_labels = labels[:split_size]
  test_windows = windows[split_size:]
  test_labels = labels[split_size:]
  train_ids = patient_ids[:split_size]
  test_ids = patient_ids[split_size:]

  return train_windows, test_windows, train_labels, test_labels, train_ids, test_ids

train_windows, test_windows, train_labels, test_labels, train_ids, test_ids = make_train_test_splits(windows, labels, patient_ids)
len(train_windows), len(test_windows), len(train_labels), len(test_labels), len(train_ids), len(test_ids)

continuous_cols = ['HR', 'O2Sat', 'MAP', 'Resp', 'ICULOS']
PATIENT_COL = 'patient_id'
LABEL_COL   = 'SepsisLabel'
FEATURE_COLS= [c for c in df.columns if c not in {PATIENT_COL, LABEL_COL}]
cont_idx = [FEATURE_COLS.index(c) for c in continuous_cols]
n_train, window_size, n_features = train_windows.shape
train_flat = train_windows.reshape(-1, n_features)
test_flat  = test_windows.reshape(-1, n_features)

scaler = RobustScaler().fit(train_flat[:, cont_idx])
train_flat[:, cont_idx] = scaler.transform(train_flat[:, cont_idx])
test_flat[:,  cont_idx] = scaler.transform(test_flat[:,  cont_idx])

train_w = train_flat.reshape(n_train, window_size, n_features)
test_w  = test_flat.reshape(test_windows.shape[0], window_size, n_features)

def lstm(window_size, n_features):

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(window_size, n_features)),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.LSTM(128),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(HORIZON, activation='sigmoid')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            'AUC',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.F1Score(
    average='micro', threshold=0.5, name='f1'
)
        ]
    )
    return model

train_labels = train_labels.reshape(-1, 1)
test_labels  = test_labels.reshape(-1, 1)

n_neg = np.sum(train_labels == 0)
n_pos = np.sum(train_labels == 1)
cw = n_neg / n_pos
print(cw)

class_weight = {
    0: 1.0,
    1: cw
}

model = lstm(
    window_size=WINDOW_SIZE,
    n_features=train_w.shape[2]
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_f1',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    train_w,
    train_labels,
    validation_data=(test_w, test_labels),
    epochs=40,
    batch_size=64,
    class_weight=class_weight,
    callbacks=[early_stop]
)

metrics = ['loss', 'accuracy', 'recall']
colors = {
    'train': 'blue',
    'val': 'red'
}
plt.figure(figsize=(10, 12))
for idx, metric in enumerate(metrics, start=1):
    plt.subplot(len(metrics), 1, idx)
    plt.plot(history.history[metric], label=f'Train {metric}', color=colors['train'])
    plt.plot(history.history[f'val_{metric}'], label=f'Val {metric}', color=colors['val'])
    plt.title(f'{metric.title()} over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel(metric.title())
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.show()

y_prob = model.predict(test_w)
y_pred = (y_prob >= 0.5).astype(int).flatten()

fpr, tpr, thresholds = roc_curve(test_labels, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--', label='Chance')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

cm = confusion_matrix(test_labels, y_pred, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax, cmap=plt.cm.Blues)
plt.title('Confusion Matrix (Threshold = 0.5)')
plt.show()

os.makedirs('labels', exist_ok=True)
os.makedirs('predictions', exist_ok=True)

y_pred = (y_prob >= 0.5).astype(int).flatten()

test_ids_1d      = np.asarray(test_ids).ravel()
test_labels_1d   = np.asarray(test_labels).ravel()
y_prod_1d        = np.asarray(y_prob).ravel()
y_pred_1d        = np.asarray(y_pred).ravel()

testing = pd.DataFrame({
    'PatientID':            test_ids_1d,
    'SepsisLabel':          test_labels_1d,
    'PredictedProbability': y_prod_1d,
    'PredictedLabel':       y_pred_1d
})

for pid, g in testing.groupby('PatientID', sort=False):
    g[['SepsisLabel']].to_csv(
        f'labels/patient_{pid}_labels.psv', sep='|', index=False)

    g[['PredictedProbability', 'PredictedLabel']].to_csv(
        f'predictions/patient_{pid}_predictions.psv', sep='|', index=False)
    

import numpy as np, os, os.path, sys, warnings

def evaluate_sepsis_score(label_directory, prediction_directory):
    # Set parameters.
    label_header       = 'SepsisLabel'
    prediction_header  = 'PredictedLabel'
    probability_header = 'PredictedProbability'

    dt_early   = -12
    dt_optimal = -6
    dt_late    = 3

    max_u_tp = 1
    min_u_fn = -2
    u_fp     = -0.05
    u_tn     = 0

    # Find label and prediction files.
    label_files = []
    for f in os.listdir(label_directory):
        g = os.path.join(label_directory, f)
        if os.path.isfile(g) and not f.lower().startswith('.') and f.lower().endswith('psv'):
            label_files.append(g)
    label_files = sorted(label_files)

    prediction_files = []
    for f in os.listdir(prediction_directory):
        g = os.path.join(prediction_directory, f)
        if os.path.isfile(g) and not f.lower().startswith('.') and f.lower().endswith('psv'):
            prediction_files.append(g)
    prediction_files = sorted(prediction_files)

    if len(label_files) != len(prediction_files):
        raise Exception('Numbers of label and prediction files must be the same.')

    # Load labels and predictions.
    num_files            = len(label_files)
    cohort_labels        = []
    cohort_predictions   = []
    cohort_probabilities = []

    for k in range(num_files):
        labels        = load_column(label_files[k], label_header, '|')
        predictions   = load_column(prediction_files[k], prediction_header, '|')
        probabilities = load_column(prediction_files[k], probability_header, '|')

        # Check labels and predictions for errors.
        if not (len(labels) == len(predictions) and len(predictions) == len(probabilities)):
            raise Exception('Numbers of labels and predictions for a file must be the same.')

        num_rows = len(labels)

        for i in range(num_rows):
            if labels[i] not in (0, 1):
                raise Exception('Labels must satisfy label == 0 or label == 1.')

            if predictions[i] not in (0, 1):
                raise Exception('Predictions must satisfy prediction == 0 or prediction == 1.')

            if not 0 <= probabilities[i] <= 1:
                warnings.warn('Probabilities do not satisfy 0 <= probability <= 1.')

        if 0 < np.sum(predictions) < num_rows:
            min_probability_positive = np.min(probabilities[predictions == 1])
            max_probability_negative = np.max(probabilities[predictions == 0])

            if min_probability_positive <= max_probability_negative:
                warnings.warn('Predictions are inconsistent with probabilities, i.e., a positive prediction has a lower (or equal) probability than a negative prediction.')

        # Record labels and predictions.
        cohort_labels.append(labels)
        cohort_predictions.append(predictions)
        cohort_probabilities.append(probabilities)

    # Compute AUC, accuracy, and F-measure.
    labels        = np.concatenate(cohort_labels)
    predictions   = np.concatenate(cohort_predictions)
    probabilities = np.concatenate(cohort_probabilities)

    auroc, auprc        = compute_auc(labels, probabilities)
    accuracy, f_measure = compute_accuracy_f_measure(labels, predictions)

    # Compute utility.
    observed_utilities = np.zeros(num_files)
    best_utilities     = np.zeros(num_files)
    worst_utilities    = np.zeros(num_files)
    inaction_utilities = np.zeros(num_files)

    for k in range(num_files):
        labels = cohort_labels[k]
        num_rows          = len(labels)
        observed_predictions = cohort_predictions[k]
        best_predictions     = np.zeros(num_rows)
        worst_predictions    = np.zeros(num_rows)
        inaction_predictions = np.zeros(num_rows)

        if np.any(labels):
            t_sepsis = np.argmax(labels) - dt_optimal
            best_predictions[max(0, t_sepsis + dt_early) : min(t_sepsis + dt_late + 1, num_rows)] = 1
        worst_predictions = 1 - best_predictions

        observed_utilities[k] = compute_prediction_utility(labels, observed_predictions, dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn)
        best_utilities[k]     = compute_prediction_utility(labels, best_predictions, dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn)
        worst_utilities[k]    = compute_prediction_utility(labels, worst_predictions, dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn)
        inaction_utilities[k] = compute_prediction_utility(labels, inaction_predictions, dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn)

    unnormalized_observed_utility = np.sum(observed_utilities)
    unnormalized_best_utility     = np.sum(best_utilities)
    unnormalized_worst_utility    = np.sum(worst_utilities)
    unnormalized_inaction_utility = np.sum(inaction_utilities)

    normalized_observed_utility = (unnormalized_observed_utility - unnormalized_inaction_utility) / (unnormalized_best_utility - unnormalized_inaction_utility)

    return auroc, auprc, accuracy, f_measure, normalized_observed_utility


def load_column(filename, header, delimiter):
    column = []
    with open(filename, 'r') as f:
        for i, l in enumerate(f):
            arrs = l.strip().split(delimiter)
            if i == 0:
                try:
                    j = arrs.index(header)
                except:
                    raise Exception('{} must contain column with header {} containing numerical entries.'.format(filename, header))
            else:
                if len(arrs[j]):
                    column.append(float(arrs[j]))
    return np.array(column)

def compute_auc(labels, predictions, check_errors=True):
    # Check inputs for errors.
    if check_errors:
        if len(predictions) != len(labels):
            raise Exception('Numbers of predictions and labels must be the same.')

        for label in labels:
            if not label in (0, 1):
                raise Exception('Labels must satisfy label == 0 or label == 1.')

        for prediction in predictions:
            if not 0 <= prediction <= 1:
                warnings.warn('Predictions do not satisfy 0 <= prediction <= 1.')

    # Find prediction thresholds.
    thresholds = np.unique(predictions)[::-1]
    if thresholds[0] != 1:
        thresholds = np.insert(thresholds, 0, 1)
    if thresholds[-1] == 0:
        thresholds = thresholds[:-1]

    n = len(labels)
    m = len(thresholds)

    # Populate contingency table across prediction thresholds.
    tp = np.zeros(m)
    fp = np.zeros(m)
    fn = np.zeros(m)
    tn = np.zeros(m)

    # Find indices that sort the predicted probabilities from largest to
    # smallest.
    idx = np.argsort(predictions)[::-1]

    i = 0
    for j in range(m):
        # Initialize contingency table for j-th prediction threshold.
        if j == 0:
            tp[j] = 0
            fp[j] = 0
            fn[j] = np.sum(labels)
            tn[j] = n - fn[j]
        else:
            tp[j] = tp[j - 1]
            fp[j] = fp[j - 1]
            fn[j] = fn[j - 1]
            tn[j] = tn[j - 1]

        # Update contingency table for i-th largest predicted probability.
        while i < n and predictions[idx[i]] >= thresholds[j]:
            if labels[idx[i]]:
                tp[j] += 1
                fn[j] -= 1
            else:
                fp[j] += 1
                tn[j] -= 1
            i += 1

    # Summarize contingency table.
    tpr = np.zeros(m)
    tnr = np.zeros(m)
    ppv = np.zeros(m)
    npv = np.zeros(m)

    for j in range(m):
        if tp[j] + fn[j]:
            tpr[j] = tp[j] / (tp[j] + fn[j])
        else:
            tpr[j] = 1
        if fp[j] + tn[j]:
            tnr[j] = tn[j] / (fp[j] + tn[j])
        else:
            tnr[j] = 1
        if tp[j] + fp[j]:
            ppv[j] = tp[j] / (tp[j] + fp[j])
        else:
            ppv[j] = 1
        if fn[j] + tn[j]:
            npv[j] = tn[j] / (fn[j] + tn[j])
        else:
            npv[j] = 1

    auroc = 0
    auprc = 0
    for j in range(m-1):
        auroc += 0.5 * (tpr[j + 1] - tpr[j]) * (tnr[j + 1] + tnr[j])
        auprc += (tpr[j + 1] - tpr[j]) * ppv[j + 1]

    return auroc, auprc

def compute_accuracy_f_measure(labels, predictions, check_errors=True):
    # Check inputs for errors.
    if check_errors:
        if len(predictions) != len(labels):
            raise Exception('Numbers of predictions and labels must be the same.')

        for label in labels:
            if not label in (0, 1):
                raise Exception('Labels must satisfy label == 0 or label == 1.')

        for prediction in predictions:
            if not prediction in (0, 1):
                raise Exception('Predictions must satisfy prediction == 0 or prediction == 1.')

    # Populate contingency table.
    n = len(labels)
    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for i in range(n):
        if labels[i] and predictions[i]:
            tp += 1
        elif not labels[i] and predictions[i]:
            fp += 1
        elif labels[i] and not predictions[i]:
            fn += 1
        elif not labels[i] and not predictions[i]:
            tn += 1

    # Summarize contingency table.
    if tp + fp + fn + tn:
        accuracy = float(tp + tn) / float(tp + fp + fn + tn)
    else:
        accuracy = 1.0

    if 2 * tp + fp + fn:
        f_measure = float(2 * tp) / float(2 * tp + fp + fn)
    else:
        f_measure = 1.0

    return accuracy, f_measure


def compute_prediction_utility(labels, predictions, dt_early=-12, dt_optimal=-6, dt_late=3.0, max_u_tp=1, min_u_fn=-2, u_fp=-0.05, u_tn=0, check_errors=True):
    # Check inputs for errors.
    if check_errors:
        if len(predictions) != len(labels):
            raise Exception('Numbers of predictions and labels must be the same.')

        for label in labels:
            if not label in (0, 1):
                raise Exception('Labels must satisfy label == 0 or label == 1.')

        for prediction in predictions:
            if not prediction in (0, 1):
                raise Exception('Predictions must satisfy prediction == 0 or prediction == 1.')

        if dt_early >= dt_optimal:
            raise Exception('The earliest beneficial time for predictions must be before the optimal time.')

        if dt_optimal >= dt_late:
            raise Exception('The optimal time for predictions must be before the latest beneficial time.')

    # Does the patient eventually have sepsis?
    if np.any(labels):
        is_septic = True
        t_sepsis = np.argmax(labels) - dt_optimal
    else:
        is_septic = False
        t_sepsis = float('inf')

    n = len(labels)

    # Define slopes and intercept points for utility functions of the form
    # u = m * t + b.
    m_1 = float(max_u_tp) / float(dt_optimal - dt_early)
    b_1 = -m_1 * dt_early
    m_2 = float(-max_u_tp) / float(dt_late - dt_optimal)
    b_2 = -m_2 * dt_late
    m_3 = float(min_u_fn) / float(dt_late - dt_optimal)
    b_3 = -m_3 * dt_optimal

    # Compare predicted and true conditions.
    u = np.zeros(n)
    for t in range(n):
        if t <= t_sepsis + dt_late:
            # TP
            if is_septic and predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = max(m_1 * (t - t_sepsis) + b_1, u_fp)
                elif t <= t_sepsis + dt_late:
                    u[t] = m_2 * (t - t_sepsis) + b_2
            # FP
            elif not is_septic and predictions[t]:
                u[t] = u_fp
            # FN
            elif is_septic and not predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = 0
                elif t <= t_sepsis + dt_late:
                    u[t] = m_3 * (t - t_sepsis) + b_3
            # TN
            elif not is_septic and not predictions[t]:
                u[t] = u_tn

    # Find total utility for patient.
    return np.sum(u)

if __name__ == '__main__':

    LABEL_DIR = '/content/labels'        # folder that holds SepsisLabel .psv
    PRED_DIR  = '/content/predictions'   # folder that holds prediction .psv

    auroc, auprc, accuracy, f_measure, utility = evaluate_sepsis_score(
        LABEL_DIR,
        PRED_DIR
    )

    output_string = (
        'AUROC|AUPRC|Accuracy|F-measure|Utility\n'
        f'{auroc}|{auprc}|{accuracy}|{f_measure}|{utility}'
    )

    OUT_FILE = None

    if OUT_FILE:
        with open(OUT_FILE, 'w') as f:
            f.write(output_string)
        print(f'Results written to {OUT_FILE}')
    else:
        print(output_string)

def utility(
    label_dir,
    pred_dir,
    hours_before_onset=range(24, -1, -1),  # e.g. from 24h before down to 0h
    threshold=0.5,
    # the utility parameters you already have:
    dt_early=-12, dt_optimal=-6, dt_late=3,
    max_u_tp=1, min_u_fn=-2, u_fp=-0.05, u_tn=0
):
    # --- 1) gather file lists
    label_files = sorted(f for f in os.listdir(label_dir) 
                         if f.endswith('.psv') and not f.startswith('.'))
    pred_files  = sorted(f for f in os.listdir(pred_dir)  
                         if f.endswith('.psv') and not f.startswith('.'))
    assert len(label_files) == len(pred_files), "Label/prediction count mismatch"

    # --- 2) load all labels & probabilities
    cohort_labels = []
    cohort_probs  = []
    for lf, pf in zip(label_files, pred_files):
        L = load_column(os.path.join(label_dir, lf),
                        'SepsisLabel', '|').astype(int)
        P = load_column(os.path.join(pred_dir, pf),
                        'PredictedProbability', '|')
        assert L.shape == P.shape
        cohort_labels.append(L)
        cohort_probs.append(P)

    # --- 3) precompute U_best and U_inaction
    U_best = 0.0
    U_inact = 0.0
    for L in cohort_labels:
        n = len(L)
        # best = predict at every beneficial hour
        best_pred = np.zeros(n, int)
        if L.any():
            i_on = np.argmax(L)
            t_opt = i_on + dt_optimal
            start = max(0, t_opt + dt_early)
            end   = min(n, t_opt + dt_late + 1)
            best_pred[start:end] = 1
        inact_pred = np.zeros(n, int)

        U_best  += compute_prediction_utility(
                       L, best_pred,
                       dt_early, dt_optimal, dt_late,
                       max_u_tp, min_u_fn, u_fp, u_tn)
        U_inact += compute_prediction_utility(
                       L, inact_pred,
                       dt_early, dt_optimal, dt_late,
                       max_u_tp, min_u_fn, u_fp, u_tn)

    # --- 4) for each lead time, compute normalized utility
    utilities = []
    for h in hours_before_onset:
        total_u = 0.0
        for L, P in zip(cohort_labels, cohort_probs):
            n = len(L)
            pred = np.zeros(n, int)
            if L.any():
                i_on = np.argmax(L)
                t_pred = i_on - h
                if 0 <= t_pred < n and P[int(t_pred)] >= threshold:
                    pred[int(t_pred)] = 1
            total_u += compute_prediction_utility(
                           L, pred,
                           dt_early, dt_optimal, dt_late,
                           max_u_tp, min_u_fn, u_fp, u_tn)
        # normalize
        utilities.append((total_u - U_inact) / (U_best - U_inact))

    # --- 5) plot
    plt.figure(figsize=(8,5))
    plt.plot(list(hours_before_onset), utilities, marker='o')
    plt.xlabel('Hours before sepsis onset')
    plt.ylabel('Normalized utility')
    plt.title(f'Utility vs lead time (threshold={threshold})')
    plt.grid(True)
    plt.gca().invert_xaxis()   # so larger lead times are on left
    plt.show()


# --- Usage example:
utility(
    label_dir='/content/labels',
    pred_dir ='/content/predictions',
    hours_before_onset=range(24, -1, -1),  # 24h → 0h
    threshold=0.5
)