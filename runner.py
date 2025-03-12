import cleaner as cl
import argparse
from class_balance import get_sepsis_labels, compute_missing_data, undersample_non_sepsis, load_data
from Removing_missing_features import get_high_missing_features

# Read path from command line
parser = argparse.ArgumentParser()
parser.add_argument('filename')
parser.add_argument('output')
args = parser.parse_args()
path = args.filename
out = args.output

# Put training set path here
training_setA = load_data(path)

# Undersample the dataset
status = get_sepsis_labels(training_setA)
missing_data = compute_missing_data(training_setA)
balanced_df = undersample_non_sepsis(training_setA, status, missing_data)

# Drop columns with too many missing features
columns_to_drop = get_high_missing_features(balanced_df, 75)
demographic_columns = ["Unit1", "Unit2"]
columns_to_drop += demographic_columns

# Create a cleaner object and clean
cleaner = cl.Cleaner("regression_cleaner", balanced_df, columns_to_drop, False)
cleaner.clean()

for patient in cleaner.cleaned_data:
    output = out + "/" + patient["patient_id"][0]
    patient.to_csv(output, sep='|', index=False)
