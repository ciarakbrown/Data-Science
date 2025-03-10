import cleaner as cl
from class_balance import get_sepsis_labels, compute_missing_data, undersample_non_sepsis, load_data



# Put training set path here
training_setA = load_data("/home/dipl0id/Documents/training_setA/training")

# Undersample the dataset
status = get_sepsis_labels(training_setA)
missing_data = compute_missing_data(training_setA)
balanced_df = undersample_non_sepsis(training_setA, status, missing_data)

cleaner = cl.Cleaner("arima_kalman_cleaner", balanced_df, "EtCO2", False)
cleaner.clean()
