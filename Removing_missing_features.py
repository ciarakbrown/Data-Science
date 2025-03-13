import zipfile
import os
import pandas as pd
import matplotlib.pyplot as plt

# Define dataset paths
datasets = [
    {"zip_path": "training_setA.zip", "extract_path": "extracted_data_A", "output_zip": "training_setA_cleaned.zip", "name": "A"},
    {"zip_path": "training_setB.zip", "extract_path": "extracted_data_B", "output_zip": "training_setB_cleaned.zip", "name": "B"}
]

def extract_data(zip_path, extract_path):
    """Extracts ZIP file contents."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

# Extract datasets
for dataset in datasets:
    extract_data(dataset["zip_path"], dataset["extract_path"])

# Define paths for extracted files
dataset_paths = {
    "A": os.path.join("extracted_data_A", "training"),
    "B": os.path.join("extracted_data_B", "training_setB")
}

def load_all_patients(dataset_path, dataset_name):
    """Loads all patient files from a dataset folder."""
    patient_files = [f for f in os.listdir(dataset_path) if f.endswith(".psv") or f.endswith(".csv")]
    all_dfs = []
    
    for file in patient_files:
        df = pd.read_csv(os.path.join(dataset_path, file), sep="|", low_memory=False)
        df["Source"] = dataset_name  # Track which dataset the file belongs to
        df["Patient_ID"] = file  # Store filename for later reassignment
        all_dfs.append(df)
    
    return pd.concat(all_dfs, ignore_index=True)

# Load all patient data from both datasets
df_A = load_all_patients(dataset_paths["A"], "A")
df_B = load_all_patients(dataset_paths["B"], "B")

# Combine all patients into one dataframe
combined_df = pd.concat([df_A, df_B], ignore_index=True)

# Compute percentage of missing values per feature
missing_percentage = combined_df.isnull().mean() * 100

# Plot missing data percentages
def plot_missing_data(missing_percentage):
    """Plots missing data percentages for all features."""
    missing_percentage = missing_percentage.sort_values()
    
    plt.figure(figsize=(7, 7))
    missing_percentage.plot(kind="barh", color="red")
    plt.axvline(x=50, color="blue", linestyle="--", linewidth=2, label="50% Missing Threshold")
    plt.xlabel("Percentage Missing")
    plt.ylabel("Features")
    plt.title("Average Missing Data Percentage per Feature (Both A & B Patients)")
    plt.legend()
    plt.show()

# Display the graph
plot_missing_data(missing_percentage)

# Drop features with more than 60% missing data
missing_threshold = 50
features_to_drop = missing_percentage[missing_percentage > missing_threshold].index.tolist()
cleaned_df = combined_df.drop(columns=features_to_drop, errors="ignore")

# Split cleaned data back into original datasets
cleaned_df_A = cleaned_df[cleaned_df["Source"] == "A"].drop(columns=["Source"])
cleaned_df_B = cleaned_df[cleaned_df["Source"] == "B"].drop(columns=["Source"])

# Define function to save cleaned patient files
def save_cleaned_data(cleaned_df, dataset_path, output_zip_path):
    """Saves cleaned patient data into files and zips them."""
    cleaned_data_path = os.path.join(dataset_path, "cleaned_data")
    os.makedirs(cleaned_data_path, exist_ok=True)
    
    for patient_id, patient_data in cleaned_df.groupby("Patient_ID"):
        patient_file_path = os.path.join(cleaned_data_path, patient_id)
        patient_data.drop(columns=["Patient_ID"], errors="ignore").to_csv(patient_file_path, sep="|", index=False)

    # Create ZIP file with cleaned data
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in os.listdir(cleaned_data_path):
            zipf.write(os.path.join(cleaned_data_path, file), arcname=file)

# Save cleaned datasets
save_cleaned_data(cleaned_df_A, dataset_paths["A"], "training_setA_cleaned.zip")
save_cleaned_data(cleaned_df_B, dataset_paths["B"], "training_setB_cleaned.zip")

print("Processing complete. Cleaned ZIP files created: training_setA_cleaned.zip, training_setB_cleaned.zip")

