from class_balance import save_patient_data, load_data
import os
import pandas as pd

# Read path from command line
absolute_path = os.path.dirname(__file__)

# Put training set path here
training_setA = os.path.join(absolute_path, "data/raw/training")
training_setB = os.path.join(absolute_path, "data/raw/training_setB")

# load all dataset
setA = load_data(training_setA)
setB = load_data(training_setB)

# set up labels for patients ages
bins = [18, 45, 60, 65, 75, 80, 90]
labels = [
    '18-44', 
    '45-59', 
    '60-64', 
    '65-74', 
    '75-79', 
    '80-89'
]

def process_age_group(df):
    # get patient age and group by boundaries
    age = df['Age'].iloc[0]
    age_group = pd.cut([age], bins=bins, right=False, labels=labels)[0]

    # replace the Age data with new label
    df['Age'] = age_group

    return df
# group all patients' ages
setA_processed = setA.groupby('patient_id', group_keys=False).apply(process_age_group)
setB_processed = setB.groupby('patient_id', group_keys=False).apply(process_age_group)

# save all processed data
save_patient_data(
    setA_processed, 
    os.path.join(absolute_path, "data/processed/training")
    )

save_patient_data(
    setA_processed, 
    os.path.join(absolute_path, "data/processed/training_setB")
    )