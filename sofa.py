from class_balance import load_data
import pandas as pd

def calculate_sofa_score(row):
    sofa_score = 0
    if row["MAP"] < 70:
        sofa_score += 1

    if 100 <= row["Platelets"] < 150:
        sofa_score += 1
    if 50 <= row["Platelets"] < 100:
        sofa_score += 2
    if 20 <= row["Platelets"] < 50:
        sofa_score += 3
    if row["Platelets"] < 20:
        sofa_score += 4

    if 1.2 <= row["Bilirubin_total"] <= 1.9:
        sofa_score += 1
    if 1.9 < row["Bilirubin_total"] <= 5.9:
        sofa_score += 2
    if 5.9 < row["Bilirubin_total"] <= 11.9:
        sofa_score += 3
    if row["Bilirubin_total"] > 11.9:
        sofa_score += 4

    if 1.2 <= row["Creatinine"] <= 1.9:
        sofa_score += 1
    if 1.9 < row["Creatinine"] <= 3.4:
        sofa_score += 2
    if 3.4 < row["Creatinine"] <= 4.9:
        sofa_score += 3
    if row["Creatinine"] > 4.9:
        sofa_score += 4

    if 40 <= row["FiO2"] < 53.3:
        sofa_score += 1
    if 26.7 <= row["FiO2"] < 40:
        sofa_score += 2
    if 13.3 <= row["FiO2"] < 26.7:
        sofa_score += 3
    if row["FiO2"] < 13.3:
        sofa_score += 4

    return sofa_score

def calculate_shock_index(row):
    return row["HR"] / row["SBP"]

data = load_data("/home/dipl0id/Documents/clean_out")
data["SOFA"] = 0
data["SI"] = 0
patients = [df for _,df in data.groupby("patient_id")]
for patient in patients:
    patient["SOFA"] = patient.apply(calculate_sofa_score, axis=1)
    patient["SI"] = patient.apply(calculate_shock_index, axis=1)

for patient in patients:
    output = "/home/dipl0id/Documents/clean_out_sofa" + "/" + patient["patient_id"].iloc[0]
    patient.to_csv(output, sep="|", index=False)
