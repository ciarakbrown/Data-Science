from class_balance import load_data

data = load_data("/home/dipl0id/Documents/cleaned_dataset")
data = [df for _,df in data.groupby("patient_id")]
j = 0
for patient in data:
    for column in patient.columns:
            for i in patient[column]:
                try:
                    if column != "HospAdmTime":
                        if i < 0:
                            j += 1
                except:
                    continue
print(j)
