import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, KFold

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score
from utility import *

seed = 123456
np.random.seed(seed)

def majority_vote_predict(y_train, n_test: int):
    # Ensure y_train is 1D array
    y_train = np.asarray(y_train).astype(int).ravel()
    majority_class = int(np.round(y_train.mean()) >= 0.5)  # works for 0/1
    return np.full(shape=(n_test,), fill_value=majority_class, dtype=int), majority_class

def init_model_logistic(model_name):
    match model_name:
        case "MajorityVote":
            model = None
            param_grid = None

        case "LogisticRegression":
            model = LogisticRegression(max_iter=5000, random_state=seed)
            param_grid = {
                "C": [0.01, 0.1, 1, 10, 100],
                "penalty": ["l2", None], #add none
                "solver": ["lbfgs", "liblinear"]
            }

        case "SVC":
            model = SVC(random_state=seed)
            param_grid = {
                "C": [0.1, 1, 10, 100],
                "kernel": ["linear", "rbf", "sigmoid"],
                "gamma": ["scale", "auto"]
            }

        case "RandomForest":
            model = RandomForestClassifier(random_state=seed)
            param_grid = {
                "n_estimators": [100, 300, 500, 750, 1000, 1200],
                "max_features": ["sqrt", "log2", 1, 0.5],
                "max_depth": [5, 10, 20, None],
                "min_samples_leaf": [1,2,4],
                "min_samples_split": [2, 5, 10],
                "max_samples": [0.9,0.8,0.7,0.6, 0.5]
            }

        case "KNN":
            model = KNeighborsClassifier()
            param_grid = {
                "n_neighbors": [3, 5, 7, 9, 11],
                "weights": ["distance", "uniform"],
                "p": [1, 2]
            }

        case "MLP":
            model = MLPClassifier(max_iter=2000, early_stopping=True, random_state=seed)
            param_grid = {
                "hidden_layer_sizes": [(), (20,), (64,), (100,), (20,20,20), (64, 64), (100, 100), (128,128)],
                "activation": ["tanh","relu", "logistic"],
                "solver": ["adam", "lbfgs"],
                "alpha": [0.0001, 0.001, 0.01, 0.1],
                "learning_rate": ["constant", "adaptive", "invscaling"],
                "learning_rate_init": [1e-4, 1e-3, 1e-2],
                "n_iter_no_change": [10,20,30],
                "validation_fraction": [0.1,0.15,0.2]
            }

    return model, param_grid


# train / tune model on training data
# default 5-fold CV, scoring = accuracy
def train_eval_model_logistic(model, param_grid, X_train, Y_train, X_test, Y_test):
    if model is None:
        # Majority vote baseline
        majority_class = int(np.round(np.mean(Y_train)) >= 0.5)
        Y_pred = np.full(len(Y_test), majority_class, dtype=int)
        acc = accuracy_score(Y_test, Y_pred)
        return None, Y_pred, acc, {"majority_class": majority_class}

    if param_grid is not None:
        cv = KFold(n_splits=5, shuffle=True, random_state=seed)
        est = GridSearchCV(model, param_grid, scoring="accuracy", cv=cv, n_jobs=-1)
        est.fit(X_train, Y_train)
        best_params = est.best_params_
        best_est = est.best_estimator_
    else:
        best_est = model
        best_est.fit(X_train, Y_train)
        best_params = None

    Y_pred = best_est.predict(X_test)
    acc = accuracy_score(Y_test, Y_pred)

    return best_est, Y_pred, acc, best_params



if __name__ == "__main__":
    results_df = pd.DataFrame(columns=["acc", "best_paras"], index=["MajorityVote", "LogisticRegression", "SVC", "RandomForest", "KNN", "MLP"])
    relationship_type = input("Enter relationship type (e.g., 'linear', 'square', 'exp', etc.): ")
    name = f"synthetic_data_{relationship_type}"
    data_list = [name]

    for data_name in data_list:
        print(f"Running tabular methods on {data_name}: ")
        X, Y = read_data(data_name)
        X_train, X_test, Y_train, Y_test = split_data(X, Y)
        

        range_Y = np.max(Y_train) - np.min(Y_train)
        if range_Y < 1e-3:
            r = np.mean(Y_train)
        else:
            r = range_Y
        
        print(f"The range of is {r}")
        prediction = {}
        
        #for model_name in ["MLP"]:
        for model_name in ["MajorityVote", "LogisticRegression", "SVC", "RandomForest", "KNN", "MLP"]:
            model, param_grid = init_model_logistic(model_name)
            est, Y_pred, acc, best_params = train_eval_model_logistic(model, param_grid, X_train, Y_train, X_test, Y_test)

            prediction[model_name] = Y_pred
            pred_df = pd.DataFrame({'Y_true': Y_test, 'Y_pred': Y_pred})

            results_df.loc[model_name] = [acc, str(best_params)]
            print(f"Model: {model_name}, acc: {acc}, Best Parameters: {best_params}")
            
        file_name = f"performance_data/{data_name}_performance.csv"
        results_df.to_csv(file_name, index=True)

