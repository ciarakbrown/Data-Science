from pandas import DataFrame
from pykalman import KalmanFilter
from sklearn.impute import KNNImputer, IterativeImputer

# Replace NaNs with the mean of that column that belongs to the
# target class. May introduce bias and overfit.
def class_mean_impute(df: DataFrame):
    target_column = df.columns[-1]
    nan_columns = df.columns[df.isna().all()]
    df.drop(columns=nan_columns, inplace=True)
    for col in df.columns[:-1]:
        df[col] = df.groupby(target_column)[col].transform(lambda x: x.fillna(x.mean()))

# Forward fill
def forward_fill(df: DataFrame):
    df.ffill(inplace=True)

# Backward fill
def backward_fill(df: DataFrame):
    df.bfill(inplace=True)

# Kalman filter
def kalman_fill(df: DataFrame):
    kf = KalmanFilter()
    df_imputed = kf.em(df).smooth(df)[0]
    return df_imputed

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
