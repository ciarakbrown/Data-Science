import cleaner as cl
from class_balance import get_sepsis_labels, compute_missing_data, undersample_non_sepsis, load_data
from Removing_missing_features import get_high_missing_features
import os

# Read path from command line


# Put training set path here
training_setA = os.path.join("home/dipl0id/Documents/training_setA/training")
training_setB = os.path.join("home/dipl0id/Documents/training_setB")

# Undersample the dataset
status = get_sepsis_labels(training_setA)
missing_data = compute_missing_data(training_setA)
balanced_df = undersample_non_sepsis(training_setA, status, missing_data)

# Drop columns with too many missing features
columns_to_drop = get_high_missing_features(balanced_df, 65)
demographic_columns = ["Unit1", "Unit2"]
columns_to_drop += demographic_columns
balanced_df = balanced_df.drop(columns=columns_to_drop)


# Create a cleaner object and clean
patients = [df for _,df in balanced_df.groupby("patient_id")]
for patient in patients:
    patient = patient.bfill().ffill()
    output = "home/dipl0id/Documents/data_out" + "/" + patient["patient_id"].iloc[0]
    patient.to_csv(output, sep="|", index=False)
