
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# read in data, specify features and target
# for all data, drop any row with NA
# if possible, normalize target to be between 0 and 1
def read_data(data_name):
    match data_name:
        case "wine-red":
            data = pd.read_csv("data/winequality-red.csv", sep=";")
            data = data.dropna()
            X = data.drop(columns=["quality"])
            Y = data["quality"] / 10
        case "wine-red-nf":
            data = pd.read_csv("data/winequality-red-nf.csv", sep=";")
            data = data.dropna()
            X = data.drop(columns=["quality"])
            Y = data["quality"] / 10
        case data_name if data_name.startswith("synthetic_"):
            data = pd.read_csv(f"data/{data_name}.csv")
            #data = data.sample(frac=1).reset_index(drop=True)
            X = data.drop(columns=["Y"])
            Y = data["Y"]
            # min-max normalize Y
            # Y = (Y - Y.min()) / (Y.max() - Y.min())
        case "laptop_noduplicates":
            data = pd.read_csv("data/laptop_noduplicates.csv")
            data = data.dropna()
            data = data.drop(columns=["Screen Size"])
            old_columns = [col for col in data.columns if col != "Price"] + ["Price"]
            new_columns = [f"X{i}" for i in range(1, 7)] + ["Y"]
            data = data.rename(columns=dict(zip(old_columns, new_columns)))
            X = data.drop(columns=["Y"])
            Y = data["Y"]
        case "multiplication":
            data = pd.read_csv("data/multiplication.csv")
            X = data.drop(columns=["c"])
            Y = data["c"]

    return X, Y

# split data into training and testing sets
# by default keep 80% training and 20% testing
# note that the 80% is later used for GridSearchCV or GPT fine-tuning
def split_data(X, Y, p = 0.8, seed = 123456):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = p, random_state = seed)
    return X_train, X_test, Y_train, Y_test

def split_outlier_data(X, Y, p = 0.8, seed = 123456):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = p, random_state = seed)
    training = pd.concat([X_train, Y_train], axis=1)
    new_train = add_outliers(training, 5) # Change the percent
    X_new_train = new_train.iloc[:, :-1]  
    Y_new_train = new_train.iloc[:, -1]   

    return X_new_train, X_test, Y_new_train, Y_test

def add_outliers(data, outlier_percentage, percentile_threshold=95, seed = 123456):
    np.random.seed(seed)
    n_outliers = int(len(data) * outlier_percentage / 100)
    outlier_indices = np.random.choice(len(data), n_outliers, replace=False)

    data_array = data.values.copy()
    
    n_features = data_array.shape[1] - 1

    # Compute percentile for X and Y
    lower_bounds_X = np.percentile(data_array[:, :-1], 100 - percentile_threshold, axis=0)
    upper_bounds_X = np.percentile(data_array[:, :-1], percentile_threshold, axis=0)
    lower_bound_Y = np.percentile(data_array[:, -1], 100 - percentile_threshold)
    upper_bound_Y = np.percentile(data_array[:, -1], percentile_threshold)

    for idx in outlier_indices:
        feature_directions = np.random.choice([-1, 1], size=n_features)  
        for col in range(n_features):
            if feature_directions[col] == -1:
                data_array[idx, col] = np.random.uniform(lower_bounds_X[col] - abs(lower_bounds_X[col] * 0.2), lower_bounds_X[col])
            else:
                data_array[idx, col] = np.random.uniform(upper_bounds_X[col], upper_bounds_X[col] + abs(upper_bounds_X[col] * 0.2))
    
    Y_directions = np.random.choice([-1, 1], size=n_outliers)  
    Y_outliers = np.where(
        Y_directions == -1, 
        np.random.uniform(lower_bound_Y - abs(lower_bound_Y * 0.2), lower_bound_Y, size=n_outliers),
        np.random.uniform(upper_bound_Y, upper_bound_Y + abs(upper_bound_Y * 0.2), size=n_outliers)
    )
    data_array[outlier_indices, -1] = Y_outliers  # Apply the Y outlier values

    col_names = [f"X{i}" for i in range(n_features)] + ["Y"]
    result_df = pd.DataFrame(data_array, columns=col_names)
    
    return result_df

def split_missing_data(X, Y, p=0.8, missing_percent=1, seed=123456):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=p, random_state=seed)
    
    training = pd.concat([X_train, Y_train], axis=1)
    new_train = add_missing_values(training, missing_percent, seed)
    
    X_new_train = new_train.iloc[:, :-1]
    Y_new_train = new_train.iloc[:, -1]
    
    return X_new_train, X_test, Y_new_train, Y_test

def add_missing_values(data, missing_percent, seed=123456):
    np.random.seed(seed)
    data_array = data.values.copy()
    
    n_rows, n_cols = data_array.shape
    n_missing = int(n_rows * (missing_percent / 100.0))
    
    # Generate flat random indices to inject NaNs
    missing_indices = np.random.choice(n_rows, size=n_missing, replace=False)

    # Get 2D index positions from flat indices
    rows, cols = np.unravel_index(missing_indices, data_array.shape)
    
    data_array[rows, cols] = np.nan
    
    return pd.DataFrame(data_array, columns=data.columns)

# report performance metrics
def performance_eval(Y_pred, Y_test):
    Y_pred = np.array(Y_pred)
    Y_test = np.array(Y_test)
    MAE = np.mean(np.abs(Y_pred - Y_test))
    RMSE = np.sqrt(np.mean((Y_pred - Y_test)**2))
    MAPE = np.mean(np.abs(Y_test - Y_pred) / ((np.abs(Y_test) + np.abs(Y_pred)) / 2)) * 100

    #accuracy = np.mean(Y_pred == Y_test) * 100

    return MAE, RMSE, MAPE
