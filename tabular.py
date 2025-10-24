import numpy as np
import pandas as pd
from utility import *
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
#import xgboost as xgb
#import lightgbm as lgb

seed = 123456
np.random.seed(seed)

# intialize tabular model and parameter grid
def init_model(model_name):
    match model_name:
        case "LinearRegression":
            model = LinearRegression()
            param_grid = None
        case "LassoRegression":
            model = Lasso(random_state=seed)
            param_grid = {"alpha": [0.01, 0.5, 1]}
        case "SVR":
            model = SVR()
            param_grid = {"C": [0.1, 1, 10, 100],
                          "epsilon": [0.1, 0.5, 1, 2],
                          "kernel": ["linear", "rbf", "sigmoid"]}
        case "RandomForest":
            model = RandomForestRegressor(random_state=seed)
            param_grid = {"n_estimators": [300, 500, 750, 1000, 1200],
                          "max_features": [1, 0.8, 0.5, 0.2],
                          "max_depth": [5, 10, 20, None],
                          "max_samples": [1, 0.8, 0.5, 0.2]}
        case "KNN":
            model = KNeighborsRegressor()
            param_grid = {
                "n_neighbors": [3, 5, 7, 9, 11],
                "weights": ["distance", "uniform"],
                "p": [1, 2]  # p=1 for Manhattan, p=2 for Euclidean
            }
        case "MLP":
            model = MLPRegressor(max_iter=5000, early_stopping=True, random_state=seed)
            param_grid = {
                "hidden_layer_sizes": [(5,), (20,), (50,), (5, 5), (10, 10), (100,100)],
                "activation": ["identity", "tanh", "relu"],
                "solver": ["adam", "sgd"],
                "alpha": [0.001, 0.005, 0.01, 0.05, 0.1],
                "learning_rate": ["constant", "adaptive"]
            }
    
    return model, param_grid

# train / tune model on training data
# default 5-fold cross-validation
# evaluate on X_test, Y_test
def train_eval_model(model, param_grid, X_train, Y_train, X_test, Y_test):
    if param_grid is not None:
        cv = KFold(n_splits=5, shuffle=True, random_state=seed)
        est = GridSearchCV(model, param_grid, scoring="neg_mean_squared_error", cv=cv)
        #est = GridSearchCV(model, param_grid, scoring = "neg_mean_squared_error", cv = 5)
        est.fit(X_train, Y_train)
        best_params = est.best_params_
    else:
        est = model
        est.fit(X_train, Y_train)
        best_params = None
    
    # evaluation, report MAE and RMSE
    Y_pred = est.predict(X_test)
    MAE, RMSE, MAPE = performance_eval(Y_pred, Y_test)

    return est, Y_pred, MAE, RMSE, MAPE, best_params

if __name__ == "__main__":
    prediction_results_df = pd.DataFrame(columns=["MAPC", "MAPC_std", "MAPC_max"], index=["LinearRegression", "LassoRegression", "SVR", "RandomForest", "KNN", "MLP"])
    results_df = pd.DataFrame(columns=["MAE", "RMSE", "MAPE", "best_paras"], index=["LinearRegression", "LassoRegression", "SVR", "RandomForest", "KNN", "MLP"])
    #relationship_type = input("Enter relationship type (e.g., 'linear', 'square', 'exp', etc.): ")
    #name = f"synthetic_data_{relationship_type}"
    data_list = [name]

    for data_name in data_list:
        print(f"Running tabular methods on {data_name}: ")
        X, Y = read_data(data_name)
        X_train, X_test, Y_train, Y_test = split_data(X, Y)
        train_df = pd.concat([X_train, Y_train], axis=1)
        train_df = train_df.dropna()
        X_train = train_df.iloc[:, :-1]
        Y_train = train_df.iloc[:, -1]


        

        range_Y = np.max(Y_train) - np.min(Y_train)
        if range_Y < 1e-3:
            r = np.mean(Y_train)
        else:
            r = range_Y
        
        print(f"The range of is {r}")
        prediction = {}
        
        for model_name in ["LinearRegression", "LassoRegression", "SVR", "RandomForest", "KNN", "MLP"]:
        #for model_name in ['MLP']:
            model, param_grid = init_model(model_name)
            est, Y_pred, MAE, RMSE, MAPE, best_params = train_eval_model(model, param_grid, X_train, Y_train, X_test, Y_test)

            prediction[model_name] = Y_pred
            pred_df = pd.DataFrame({'Y_true': Y_test, 'Y_pred': Y_pred})

            results_df.loc[model_name] = [MAE, RMSE, MAPE, str(best_params)]
            print(f"Model: {model_name}, MAE: {MAE}, RMSE: {RMSE}, MAPE: {MAPE}, Best Parameters: {best_params}")
            
        file_name = f"performance_data/{data_name}_performance.csv"
        #prediction_file_name = f"performance_data/{data_name}_prediction_performance.csv"
        results_df.to_csv(file_name, index=True)
        #prediction_results_df.to_csv(prediction_file_name, index=True)
        #print(f"Best parameters: {est.best_params_}")


    
