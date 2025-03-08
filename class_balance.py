import os
import pandas as pd
import glob
import numpy as np


def load_data(path):
    """
    Loads patient data from PSV files and returns the full dataset with patient IDs column added.
    """
    files = glob.glob(os.path.join(path, '*.psv'))
    data_frames = []

    for file in files:
        df = pd.read_csv(file, sep='|')
        patient_id = os.path.basename(file)  # Use the file name as patient ID
        df['patient_id'] = patient_id  # Add patient_id column
        data_frames.append(df)

    full_dataset = pd.concat(data_frames, ignore_index=True)
    return full_dataset


def get_sepsis_labels(df):
    """Determines which patients have sepsis (SepsisLabel=1 in any row)."""
    sepsis_status = df.groupby('patient_id')['SepsisLabel'].max()
    return sepsis_status


def compute_missing_data(df):
    """Computes missing data percentage per patient."""
    missing_data = df.isnull().groupby(df['patient_id']).mean()  # Mean of NaNs per column per patient
    missing_data['missing_percentage'] = missing_data.mean(axis=1)  # Average missing percentage per patient
    return missing_data[['missing_percentage']]


def undersample_non_sepsis(df, sepsis_status, missing_data):
    """Removes non-sepsis patients with the highest missing data first to balance dataset."""

    patient_info = pd.DataFrame({'SepsisLabel': sepsis_status}).merge(missing_data, on='patient_id')
    sepsis_patients = patient_info[patient_info['SepsisLabel'] == 1]
    non_sepsis_patients = patient_info[patient_info['SepsisLabel'] == 0]

    # Sort non-sepsis patients by missing percentage (highest missing data first)
    non_sepsis_patients = non_sepsis_patients.sort_values(by='missing_percentage', ascending=False)

    # Balance the dataset and create new
    num_sepsis = len(sepsis_patients)
    balanced_non_sepsis = non_sepsis_patients.iloc[:num_sepsis]  # Keep only as many as there are sepsis patients
    balanced_patient_ids = set(sepsis_patients.index).union(set(balanced_non_sepsis.index))
    balanced_df = df[df['patient_id'].isin(balanced_patient_ids)]

    return balanced_df


# Obtain full dataset with added patient id column
path = '/Users/ciarabrown/Documents/DS/training'
full_data = load_data(path)

# List of patient id and either a 1 or 0
sepsis_status = get_sepsis_labels(full_data)
print(sepsis_status)

# List of patient id and % of missing data that patient has
missing_data = compute_missing_data(full_data)
print(missing_data)

balanced_df = undersample_non_sepsis(full_data, sepsis_status, missing_data)
print(balanced_df)

# Show class distribution after under-sampling
print(balanced_df.groupby('patient_id')['SepsisLabel'].max().value_counts())


# -----------------------------save the data into----------------------------------------

def save_patient_data(df, output_path):
    """Saves each patient's data in a separate .psv file, maintaining the original format."""
    os.makedirs(output_path, exist_ok=True)  # Ensure output directory exists

    for patient_id, patient_data in df.groupby('patient_id'):
        patient_data = patient_data.drop(columns=['patient_id'])  # Remove patient_id before saving
        file_path = os.path.join(output_path, patient_id)  # Use original filename format
        patient_data.to_csv(file_path, sep='|', index=False)

# Save processed data per patient
output_path = '/Users/ciarabrown/Documents/DS/balanced_data'
save_patient_data(balanced_df, output_path)