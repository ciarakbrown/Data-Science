import zipfile
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Function to extract ZIP files
def extract_data(zip_path, extract_path):
    """Extracts the ZIP file and returns the correct training directory path."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    return os.path.join(extract_path, "training_setB")

# Function to load patient data
def load_patient_data(training_path, num_files=100):
    """Loads patient data from the extracted CSV files."""
    files = os.listdir(training_path)[:num_files]
    data_list = []
    for file in files:
        file_path = os.path.join(training_path, file)
        df = pd.read_csv(file_path, sep="|")
        df["Patient_ID"] = file.split(".")[0]
        data_list.append(df)

    return pd.concat(data_list, ignore_index=True)

# Function to generate missing data heatmap
def plot_missing_data_heatmap(patient_data_b):
    """Plots a heatmap showing missing data proportions, sorted by patient age with labeled age group transitions."""

    # Aggregate missing data proportion per patient
    patient_missing_proportion = patient_data_b.groupby("Patient_ID").apply(lambda df: df.isnull().mean())

    # Sort patients by Age (using the first recorded Age per Patient_ID)
    patient_age_mapping = patient_data_b.groupby("Patient_ID")["Age"].first()
    sorted_patients = patient_age_mapping.sort_values().index

    # Reorder the missing data proportion dataframe by age
    patient_missing_proportion = patient_missing_proportion.loc[sorted_patients]

    # Define age bins (10-year intervals)
    bins = list(range(0, 110, 10))  # 0-9, 10-19, ..., 100+
    labels = [f"{b}-{b+9}" for b in bins[:-1]] + ["100+"]

    # Assign patients to age bins
    patient_age_binned = pd.cut(patient_age_mapping, bins=bins + [200], labels=labels, right=False)

    # Identify where the age group changes
    sorted_ages = patient_age_mapping.loc[sorted_patients]
    age_bin_edges = np.digitize(sorted_ages, bins)
    unique_bins, bin_indices = np.unique(age_bin_edges, return_index=True)
    age_labels = [labels[b - 1] for b in unique_bins]

    # Generate heatmap
    plt.figure(figsize=(12, 6))
    ax = sns.heatmap(patient_missing_proportion, cbar=True, cmap="viridis")

    # Set y-axis labels at points where the age group changes
    plt.yticks(ticks=bin_indices + 0.5, labels=age_labels, rotation=0)

    # Label the colorbar to indicate missing data levels
    cbar = ax.collections[0].colorbar
    cbar.set_label("Proportion of Missing Data (Yellow = 100% Missing)", fontsize=12)

    # Titles and labels
    plt.title("Proportion of Missing Data per Feature - Patients Ordered by Age (Dataset B)", fontsize=15)
    plt.xlabel("Feature", fontsize=14)
    plt.ylabel("Age Group (Years)", fontsize=14)
    plt.show()

# Main function to execute the workflow
def main():
    """Runs the full pipeline: extract, load, preprocess, and visualize data."""
    zip_path = "training_setB.zip"  # Dataset B path
    extract_path = "/mnt/data/extracted_data"

    # Extract files
    training_path = extract_data(zip_path, extract_path)

    # Load data
    patient_data_b = load_patient_data(training_path)

    # Generate heatmap
    plot_missing_data_heatmap(patient_data_b)

# Run the workflow
main()
