import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from class_balance import load_data


def evaluate_model(y_true,y_pred):
  accuracy = accuracy_score(y_true, y_pred)
  print("Accuracy:", accuracy)
  precision = precision_score(y_true, y_pred)
  print("Precision:", precision)
  aps = average_precision_score(y_true, y_pred)
  print("Average precision score:",  aps)
  recall = recall_score(y_true, y_pred)
  print("Recall:", recall)
  f1 = f1_score(y_true, y_pred)
  print("F1 Score:", f1)
  auc = roc_auc_score(y_true, y_pred)
  print("AUC-ROC:", auc)
  mae = mean_absolute_error(y_true, y_pred)
  print("Mean Absolute Error:", mae)
  rmse = np.sqrt(mean_squared_error(y_true, y_pred))
  print("Root Mean Squared Error:", rmse)



data = load_data("/home/dipl0id/Documents/cleaned_dataset")
data = data.drop(columns="patient_id")





positive_class = data[data["SepsisLabel"]==1]
negative_class = data[data["SepsisLabel"]==0]

negative_class_subset = negative_class.sample(n=2*len(positive_class))
data_full = pd.concat([positive_class, negative_class_subset])

positive_class = data_full[data_full["SepsisLabel"]==1]
negative_class = data_full[data_full["SepsisLabel"]==0]



X = data_full.drop("SepsisLabel", axis=1)
y = data_full["SepsisLabel"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


ratio = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
xtrn = X_train.values
xtst = X_test.values
ytrn = y_train.values
ytst = y_test.values
dtrain = xgb.DMatrix(xtrn, label=ytrn)
dtest = xgb.DMatrix(xtst, label=ytst)
param = {
    'max_depth': 5,
    'eta': 0.3,
    'silent': 1,
    'objective': 'binary:logistic'
}
num_round = 100
bst = xgb.train(param, dtrain, num_round)
pred = bst.predict(dtest)
prediction = []
for i in pred:
  if i<0.5:
    prediction.append(0)
  else:
    prediction.append(1)

evaluate_model(ytst, prediction)
