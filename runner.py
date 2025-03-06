import cleaner
import loader

# Put training set path here
training_setA = loader.loader("/home/dipl0id/Documents/training_setA/training")
df = training_setA[0]
df = df.drop(columns=["EtCO2"])
for column in df.columns:
    loc = df.columns.get_loc(column)
    x = cleaner.arima_kalman_imputer(df.iloc[:,loc])
    df[column] = x
    print(loc, " columns completed.")
print(df)
