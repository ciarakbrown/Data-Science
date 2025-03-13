from pandas import DataFrame
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from pmdarima import auto_arima
from scipy.stats import multivariate_normal
import statsmodels.api as sm

def nearest_positive_definite(A):
    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)
    H = np.dot(V.T, np.dot(np.diag(s), V))
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2
    if is_positive_definite(A3):
        return A3
    spacing = np.spacing(np.linalg.norm(A))
    identity = np.eye(A.shape[0])
    k = 1
    while not is_positive_definite(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += identity * (-mineig * k**2 + spacing)
        k += 1
    return A3

def is_positive_definite(B):
    try:
        np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False

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
        self.columns_to_ignore = ["patient_id", "SepsisLabel"]
        self.cleaned_data = []

    def clean(self):
        initial_clean = self.patient_data.drop(columns=self.columns_to_drop)
        if (not self.global_clean):
            self.patient_list = [df for _,df in initial_clean.groupby("patient_id")]
        cleaner = getattr(self, self.cleaning_method)
        cleaner()

    def mcmc_helper(self):
        mean_features_list = []
        for patient_df in self.patient_list:

            patient_df_filtered = patient_df.drop(columns=self.columns_to_ignore, errors='ignore')
            patient_mean = patient_df_filtered.mean()
            mean_features_list.append(patient_mean)
        mean_features = pd.DataFrame(mean_features_list)


        mean_vector = mean_features.mean(axis=0).values
        cov_matrix = mean_features.cov().values

        cov_matrix = nearest_positive_definite(cov_matrix)

        try:
            mvn = multivariate_normal(mean=mean_vector, cov=cov_matrix, allow_singular=True)
        except np.linalg.LinAlgError:
            cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6
            cov_matrix = nearest_positive_definite(cov_matrix)
            mvn = multivariate_normal(mean=mean_vector, cov=cov_matrix, allow_singular=True)

        mcmc_samples = mvn.rvs(size=100000)

        for patient_idx, patient_row in mean_features.iterrows():
            patient_means = patient_row.copy()
            nan_vars = patient_means.index[patient_means.isna()].tolist()

            for var in nan_vars:
                non_nan_vars = patient_means.index[~patient_means.isna()].tolist()
                non_nan_vars = [v for v in non_nan_vars if v != var]

                if not non_nan_vars:
                    var_idx = mean_features.columns.get_loc(var)
                    imputed_value = mcmc_samples[:, var_idx].mean()
                else:
                    var_indices = [mean_features.columns.get_loc(v) for v in non_nan_vars]
                    patient_non_nan_values = patient_means[non_nan_vars].values.reshape(1, -1)
                    sample_values = mcmc_samples[:, var_indices]

                    mask = np.all(
                        (sample_values >= (patient_non_nan_values - 1)) &
                        (sample_values <= (patient_non_nan_values + 1)),
                        axis=1
                    )
                    relevant_samples = mcmc_samples[mask]

                    var_idx = mean_features.columns.get_loc(var)
                    if len(relevant_samples) > 0:
                        imputed_value = relevant_samples[:, var_idx].mean()
                    else:
                        imputed_value = mcmc_samples[:, var_idx].mean()

                patient_means[var] = imputed_value
                if var not in self.columns_to_ignore:
                    self.patient_list[patient_idx][var] = self.patient_list[patient_idx][var].fillna(imputed_value)

    def arima_kalman_cleaner(self):
        for patient in self.patient_list:
            clean_df = DataFrame(columns=self.patient_list[0].columns)
            for column in patient.columns:
                loc = patient.columns.get_loc(column)
                ts = patient.iloc[:,loc]
                if column in self.columns_to_ignore:
                    clean_df[column] = ts
                    continue
                if ts.isna().all():
                    print(f"All NaN in column {column}. Skipping...")
                    clean_df[column] = ts
                    continue
                if ts.isna().sum() / len(ts) > 0.5:
                    print(f"High NaN ratio in {column}. Using alternate cleaning method...")
                    clean_df[column] = ts.bfill().ffill()
                    continue
                try:
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
                        initialization="approximate_diffuse",
                        initial_variance=1e-3
                    )
                    result = arima_ssm.fit(disp=False, maxiter=200, method="lbfgs")
                    data_filled = result.get_prediction().predicted_mean
                    ts_filled = ts.copy()
                    ts_filled[nan_mask] = data_filled[nan_mask]

                except ZeroDivisionError:
                    print("Unexpected error. Using alternate cleaning method...")
                    ts_filled = ts.bfill().ffill()
                clean_df[column] = ts_filled.bfill()
            print(clean_df)
            cleaned_data = self.cleaned_data.copy()
            cleaned_data.append(clean_df)
            self.cleaned_data = cleaned_data

    # Regression imputation
    def regression_cleaner(self):
        self.mcmc_helper()
        for patient in self.patient_list:
            clean_df = patient.drop(columns=self.columns_to_ignore)
            rf = IterativeImputer(keep_empty_features=True, max_iter=20)
            df_imputed = rf.fit_transform(clean_df)

            df_imputed = DataFrame(df_imputed, columns=clean_df.columns)
            df_imputed[self.columns_to_ignore] = patient[self.columns_to_ignore].values.tolist()

            self.cleaned_data.append(df_imputed)


    # KNN Imputation
    def knn_cleaner(self):
        for patient in self.patient_list:
            print(patient)
            knn = KNNImputer()
            df_imputed = knn.fit_transform(patient)
            cleaned_data = self.cleaned_data.copy()
            cleaned_data.append(df_imputed)
            self.cleaned_data = cleaned_data






# Replace NaNs with the mean of that column that belongs to the
# target class. May introduce bias and overfit.
def class_mean_impute(df: DataFrame):
    target_column = df.columns[-1]
    nan_columns = df.columns[df.isna().all()]
    df.drop(columns=nan_columns, inplace=True)
    for col in df.columns[:-1]:
        df[col] = df.groupby(target_column)[col].transform(lambda x: x.fillna(x.mean()))
        
    return df

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
