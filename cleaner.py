from pandas import DataFrame
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

# Drop columns that have been decided to be ignored
def base_clean(df: DataFrame, columns):
    df.drop(columns=columns)
    pass

# Replace NaNs with the mean of that column that belongs to the
# target class. May introduce bias and overfit.
def class_mean_impute(df: DataFrame):
    target_column = df.columns[-1]
    nan_columns = df.columns[df.isna().all()]
    df.drop(columns=nan_columns, inplace=True)
    for col in df.columns[:-1]:
        df[col] = df.groupby(target_column)[col].transform(lambda x: x.fillna(x.mean()))


def arima_kalman_imputer(ts):
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
    return ts_filled

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
