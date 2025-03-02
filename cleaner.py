from pandas import DataFrame
from pykalman import KalmanFilter
import numpy as np

# Replace NaNs with the mean of that column that belongs to the
# target class. May introduce bias and overfit.
def class_mean_impute(df: DataFrame):
    target_column = df.columns[-1]
    nan_columns = df.columns[df.isna().all()]
    df.drop(columns=nan_columns, inplace=True)
    for col in df.columns[:-1]:
        df[col] = df.groupby(target_column)[col].transform(lambda x: x.fillna(x.mean()))

# Kalman filter
# Simply fill NaNs with ffill and bfill first
# Apply Kalman Filter to that filled set
# Impute the NaNs with the corresponding means from the previous step
def kalman_fill(df: DataFrame):
    nan_columns = df.columns[df.isna().all()]
    df = df.drop(columns=nan_columns)
    df_temp_fill = df.ffill().bfill()
    n_dim_obs = df.shape[1]
    kf = KalmanFilter(
        initial_state_mean=df_temp_fill.iloc[0].values,
        n_dim_obs=n_dim_obs,
        initial_state_covariance=np.eye(n_dim_obs),
        observation_covariance=np.eye(n_dim_obs),
        transition_covariance=np.eye(n_dim_obs) * 0.01
    )
    state_means, _ = kf.em(df_temp_fill).smooth(df_temp_fill.values)
    df_filled = df.copy()
    for col in range(n_dim_obs):
        nan_mask = df.iloc[:, col].isna()
        df_filled.iloc[nan_mask, col] = state_means[nan_mask, col]
    return df_filled

# Regression imputation

# KNN Imputation

# Rolling mean

# Interpolation
