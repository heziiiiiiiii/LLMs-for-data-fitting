
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# read in data, specify features and target
def read_data(data_name):
    match data_name:
        case data_name if data_name.startswith("synthetic_"):
            data = pd.read_csv(f"data/{data_name}.csv")
            X = data.drop(columns=["Y"])
            Y = data["Y"]
    return X, Y

# split data into training and testing sets
# by default keep 80% training and 20% testing
# note that the 80% is later used for GridSearchCV or GPT fine-tuning
def split_data(X, Y, p = 0.8, seed = 123456):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = p, random_state = seed)
    return X_train, X_test, Y_train, Y_test

# report performance metrics
def performance_eval(Y_pred, Y_test):
    Y_pred = np.array(Y_pred)
    Y_test = np.array(Y_test)
    MAE = np.mean(np.abs(Y_pred - Y_test))
    RMSE = np.sqrt(np.mean((Y_pred - Y_test)**2))
    MAPE = np.mean(np.abs(Y_test - Y_pred) / ((np.abs(Y_test) + np.abs(Y_pred)) / 2)) * 100
    return MAE, RMSE, MAPE


