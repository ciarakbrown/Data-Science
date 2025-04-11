import zipfile
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# Path to the zip file
zip_path = "/content/cleaned_dataset (1).zip"

# Read all .psv files from the zip
all_data = []
with zipfile.ZipFile(zip_path, 'r') as z:
    for filename in z.namelist():
        if filename.endswith(".psv"):
            with z.open(filename) as f:
                df = pd.read_csv(f, sep='|')
                all_data.append(df)

# Combine into one DataFrame
full_df = pd.concat(all_data, ignore_index=True)

# Prepare features and labels
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
report = classification_report(y_test, y_pred, output_dict=True)
print("📊 Random Baseline Model Evaluation:")
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