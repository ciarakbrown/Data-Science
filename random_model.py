import os
import zipfile
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay,recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Path to the zip file
zip_path = "/content/cleaned_dataset (1).zip"

# STEP 1: Unzip and load data
all_data = []
with zipfile.ZipFile(zip_path, 'r') as z:
    for filename in z.namelist():
        if filename.endswith(".psv"):
            with z.open(filename) as f:
                df = pd.read_csv(f, sep='|')
                df["patient_id"] = os.path.basename(filename).replace(".psv", "")
                all_data.append(df)

full_df = pd.concat(all_data, ignore_index=True)

# Remove SBP and DBP columns
full_df = full_df.drop(columns=["SBP", "DBP"], errors="ignore")

# STEP 2: Prepare features and labels
X = full_df.drop(columns=["SepsisLabel", "patient_id", "index"], errors="ignore")
y = full_df["SepsisLabel"]
X = X.fillna(X.mean())  # Fill missing values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)

# Train random baseline model
model = DummyClassifier(strategy="uniform", random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Get classification report
print("📊 Random Baseline Model Evaluation:")
report = classification_report(y_test, y_pred, output_dict=True)
print(classification_report(y_test, y_pred, digits=4))

# Convert report to DataFrame for plotting
report_df = pd.DataFrame(report).transpose().drop("accuracy", errors="ignore")

# Plot classification metrics
plt.figure(figsize=(10, 6))
report_df[['precision', 'recall', 'f1-score']].plot(kind='bar')
plt.title("Classification Report - Random Baseline")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Optional: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Sepsis", "Sepsis"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Random Baseline")
plt.grid(False)
plt.show()

# STEP 3: Create label and prediction folders for the prediction output
os.makedirs("/content/labels", exist_ok=True)
os.makedirs("/content/predictions", exist_ok=True)

grouped = full_df.groupby("patient_id")
for patient_id, group in grouped:
    # Save label file
    label_df = group[["SepsisLabel"]]
    label_df.to_csv(f"/content/labels/{patient_id}.psv", sep='|', index=False)

    # Prepare features
    X_patient = group.drop(columns=["SepsisLabel", "patient_id", "index"], errors="ignore")
    X_patient = X_patient.fillna(X.mean())

    # Predict
    probs = model.predict_proba(X_patient)[:, 1]
    preds = model.predict(X_patient)

    pred_df = pd.DataFrame({
        "PredictedProbability": probs,
        "PredictedLabel": preds
    })
    pred_df.to_csv(f"/content/predictions/{patient_id}.psv", sep='|', index=False)

# STEP 4: Import and evaluate using PhysioNet evaluation script
sys.path.append("/content")  # Add path so we can import the script
import evaluate_sepsis_score

label_path = "/content/labels"
prediction_path = "/content/predictions"

from sklearn.metrics import recall_score

# Existing code
auroc, auprc, acc, f1, utility = evaluate_sepsis_score.evaluate_sepsis_score(label_path, prediction_path)

# Compute recall separately using full_df and the saved predictions
all_recalls = []

for patient_id, group in grouped:
    true_labels = group["SepsisLabel"].values

    pred_df = pd.read_csv(f"/content/predictions/{patient_id}.psv", sep='|')
    preds = pred_df["PredictedLabel"].values

    if len(set(true_labels)) > 1:  # Avoid division by zero errors
        recall = recall_score(true_labels, preds)
        all_recalls.append(recall)

# Average recall over all patients
avg_recall = sum(all_recalls) / len(all_recalls) if all_recalls else 0

# Print results
print("\n📊 PhysioNet Evaluation Results:")
print(f"AUROC: {auroc:.4f}")
print(f"AUPRC: {auprc:.4f}")
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Recall: {avg_recall:.4f}")
print(f"Utility Score: {utility:.4f}")

