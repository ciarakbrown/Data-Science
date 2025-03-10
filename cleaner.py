from pandas import DataFrame
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from pmdarima import auto_arima
import statsmodels.api as sm


class Cleaner:
    def __init__(self, cleaning_method, patient_data, columns_to_drop, global_clean):
        self.cleaning_methods = ["arima_kalman_cleaner",
                                "knn_cleaner",
                                "regression_cleaner",
                                "interpolation_cleaner",
                                "mean_cleaner"]
        self.global_cleaning_methods = ["mean_cleaner"]


        if cleaning_method not in self.cleaning_methods:
            raise ValueError(f"Not a valid cleaning method. Select from {self.cleaning_methods}")
        if global_clean and cleaning_method not in self.global_cleaning_methods:
            raise ValueError("global_clean set to True but a non-global cleaning method was provided.")

        self.cleaning_method = cleaning_method
        self.patient_data = patient_data
        self.global_clean = global_clean

        self.columns_to_drop = columns_to_drop
        self.cleaned_data = []

    def clean(self):
        self.cleaned_data = self.patient_data.drop(columns=self.columns_to_drop)
        if (not self.global_clean):
            self.patient_list = [df for _,df in self.cleaned_data.groupby("patient_id")]

        cleaner = getattr(self, self.cleaning_method)
        cleaner()

    def arima_kalman_cleaner(self):
        for patient in self.patient_list:
            clean_df = DataFrame(columns=self.patient_data.columns)
            for column in patient.columns:
                loc = patient.columns.get_loc(column)
                ts = patient.iloc[:,loc]
                nan_mask = ts.isna()
                ts_clean = ts.dropna()
                model = auto_arima(
                    ts_clean,
                    seasonal=False,
                    suppress_warnings=True,
                    stepwise=False,
                    information_criterion='aicc',
                    trend='ct',
                )
                arima_ssm = sm.tsa.statespace.SARIMAX(
                    ts,
                    order=model.order,
                    trend='ct',
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                    initialization="stationary",
                    initial_variance=1e4
                )
                result = arima_ssm.fit(disp=False, maxiter=200, method="nm")
                data_filled = result.get_prediction().predicted_mean
                ts_filled = ts.copy()
                ts_filled[nan_mask] = data_filled[nan_mask]
                clean_df[column] = ts_filled
            cleaned_data = self.cleaned_data.copy()
            cleaned_data.append(clean_df)
            self.cleaned_data = cleaned_data





# Replace NaNs with the mean of that column that belongs to the
# target class. May introduce bias and overfit.
def class_mean_impute(df: DataFrame):
    target_column = df.columns[-1]
    nan_columns = df.columns[df.isna().all()]
    df.drop(columns=nan_columns, inplace=True)
    for col in df.columns[:-1]:
        df[col] = df.groupby(target_column)[col].transform(lambda x: x.fillna(x.mean()))




# Regression imputation
def regression_fill(df: DataFrame):
    rf = IterativeImputer()
    df_imputed = rf.fit_transform(df)
    return df_imputed

# KNN Imputation
def knn_fill(df: DataFrame):
    knn = KNNImputer()
    df_imputed = knn.fit_transform(df)
    return df_imputed

# Rolling mean

# Interpolation
