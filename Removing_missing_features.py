'''
Looks at average percentage of missing datapoints per feature over all patients and removes features with an average of >90% missing.
'''
import zipfile
import os
import pandas as pd
import shutil
import matplotlib.pyplot as plt



def extract_data_fixed(zip_path, extract_path):
    """Extracts the ZIP file and returns the correct training directory path."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    # Detect the correct extracted folder name dynamically
    extracted_folders = [name for name in os.listdir(extract_path) if os.path.isdir(os.path.join(extract_path, name))]

    if not extracted_folders:
        raise FileNotFoundError(f"No extracted directories found in {extract_path}")

    training_path = os.path.join(extract_path, extracted_folders[0])
    print(f"Extracted to: {training_path}")  # Debugging: Print actual extracted path
    return training_path

def get_high_missing_features(df, threshold=90):
    """Identifies features with more than `threshold`% missing data."""
    missing_percent = df.isnull().mean() * 100
    return missing_percent[missing_percent > threshold].index.tolist()

def plot_missing_data(df, dataset_name):
    """Plots missing data percentage and highlights features above 90% missing."""
    missing_percent = df.isnull().mean() * 100
    missing_percent = missing_percent.sort_values()

    # Features with less than 90% missing
    features_below_threshold = missing_percent[missing_percent < 90].index.tolist()
    print(f"\nDataset: {dataset_name}")
    print(f"Features with <90% missing: {len(features_below_threshold)}")
    print(features_below_threshold)

    # Plot missing data
    plt.figure(figsize=(10, 5))
    missing_percent.plot(kind="barh", color="red")
    plt.axvline(x=90, color="blue", linestyle="--", linewidth=2, label="90% Missing Threshold")
    plt.xlabel("Percentage Missing")
    plt.ylabel("Features")
    plt.title(f"Missing Data Percentage per Feature - {dataset_name}")
    plt.legend()
    plt.show()

def clean_and_save_data_fixed(original_zip_path, extract_path, output_zip_path, dataset_name, threshold=90, max_files=5):
    """Extracts patient files, removes high-missing features per dataset, and creates a new ZIP."""

    # Step 1: Extract the ZIP file
    training_path = extract_data_fixed(original_zip_path, extract_path)
    cleaned_path = os.path.join(extract_path, "cleaned_data")
    os.makedirs(cleaned_path, exist_ok=True)

    # Step 2: Check if there are any patient files to process
    patient_files = [f for f in os.listdir(training_path) if f.endswith(".psv") or f.endswith(".csv")]
    if not patient_files:
        raise ValueError(f" No patient files found in {training_path}. Check if the ZIP file contains valid data.")

    # Step 3: Determine features to drop using a subset of files
    sample_files = patient_files[:max_files]
    combined_df = pd.concat([pd.read_csv(os.path.join(training_path, f), sep="|", low_memory=False) for f in sample_files], ignore_index=True)

    # Plot missing data before cleaning
    plot_missing_data(combined_df, dataset_name)

    features_to_drop = get_high_missing_features(combined_df, threshold)

    # Print the features being removed
    print(f"\nDataset: {dataset_name}")
    print(f"Removing {len(features_to_drop)} features with > {threshold}% missing data:")
    print(features_to_drop)

    # Step 4: Process and clean only a subset of files to avoid timeouts
    processed_files = 0
    for file in patient_files:
        if processed_files >= max_files:
            break  # Stop processing after a limited number of files

        file_path = os.path.join(training_path, file)
        df = pd.read_csv(file_path, sep="|", low_memory=False)

        # Drop high-missing features
        df_cleaned = df.drop(columns=features_to_drop, errors="ignore")

        # Save cleaned file
        cleaned_file_path = os.path.join(cleaned_path, file)
        df_cleaned.to_csv(cleaned_file_path, sep="|", index=False)

        processed_files += 1

    # Step 5: Create a new ZIP with cleaned data
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in os.listdir(cleaned_path):
            zipf.write(os.path.join(cleaned_path, file), arcname=file)

    print(f" Cleaned dataset saved as: {output_zip_path}")


if __name__ == "__main__":
    # Define the correct file paths in Colab
    datasets_fixed = [
        {"zip_path": "training_setA.zip", "extract_path": "/mnt/data/extracted_data_A", "output_zip": "/mnt/data/training_setA_cleaned.zip", "name": "Training Set A"},
        {"zip_path": "training_setB.zip", "extract_path": "/mnt/data/extracted_data_B", "output_zip": "/mnt/data/training_setB_cleaned.zip", "name": "Training Set B"}
    ]
    # Run the complete workflow for both datasets with the corrected extraction function
    for dataset in datasets_fixed:
        clean_and_save_data_fixed(dataset["zip_path"], dataset["extract_path"], dataset["output_zip"], dataset["name"], threshold=90, max_files=5)

    # Provide cleaned dataset ZIP file paths for download
    datasets_fixed[0]["output_zip"], datasets_fixed[1]["output_zip"]
