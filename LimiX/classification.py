from functools import partial
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.utils import shuffle
from huggingface_hub import hf_hub_download
try:
    from sklearn.metrics import root_mean_squared_error as mean_squared_error
except:
    from sklearn.metrics import mean_squared_error
    mean_squared_error = partial(mean_squared_error, squared=False)
import os, sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from inference.predictor import LimiXPredictor
from pathlib import Path
from utility import *
from sklearn.metrics import accuracy_score
import json

def performance_eval(Y_pred, Y_test):
    Y_pred = np.array(Y_pred)
    Y_test = np.array(Y_test)
    MAE = np.mean(np.abs(Y_pred - Y_test))
    RMSE = np.sqrt(np.mean((Y_pred - Y_test)**2))
    acc = accuracy_score(Y_test, Y_pred)

    return MAE, RMSE, acc

relationship_type = input("Enter relationship type (e.g., linear, square): ")
DATA_DIR = Path(f"Path/{relationship_type}")
k = int(input("Enter the number of examples (k) to use for few-shot learning: "))

rows = []
files = sorted(DATA_DIR.glob("dataset_*.csv"))[:100]
for csv_path in files:
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["Y"])
    Y = df["Y"]
    X_train, X_test, Y_train, Y_test = split_data(X, Y)
    if "rowOrder" in relationship_type:
        if k==10:
            X_train = X_train.iloc[:10,:]
            Y_train = Y_train[:10]
            X_train = X_train.iloc[::-1]
            Y_train = Y_train.iloc[::-1]
        elif k==20:
            X_train = X_train.iloc[:20,:]
            Y_train = Y_train[:20]
            X_train = X_train.iloc[::-1]
            Y_train = Y_train.iloc[::-1]
        elif k==500:
            X_train = X_train.iloc[:500,:]
            Y_train = Y_train[:500]
            X_train = X_train.iloc[::-1]
            Y_train = Y_train.iloc[::-1]
        else:
            X_train = X_train.iloc[::-1]
            Y_train = Y_train.iloc[::-1]

    data_device = f'cuda:0'
    model_path = hf_hub_download(repo_id="stableai-org/LimiX-16M", filename="LimiX-16M.ckpt", local_dir=".")
    model = LimiXPredictor(device=torch.device('cuda'), model_path=model_path, inference_config='config/cls_default_noretrieval.json', seed=123456)
  
    y_pred_prob = model.predict(X_train, Y_train, X_test)    
    y_pred = np.argmax(y_pred_prob, axis=1)

    MAE, RMSE, acc = performance_eval(y_pred, Y_test)

    rows.append({"csv": str(csv_path), "mae": MAE, "rmse": RMSE, "accuracy": acc, "predictions": json.dumps([int(p) for p in y_pred])})

df = pd.DataFrame(rows)
DATA_DIR_output = Path(Path)
if k==4000:
    df.to_csv(DATA_DIR_output / f"results_{relationship_type}.csv", index=False)
    df[["mae", "rmse", "accuracy"]].describe().to_csv(DATA_DIR_output / f"summary_statistics_{relationship_type}.csv")
else:
    df.to_csv(DATA_DIR_output / f"results_{relationship_type}_fewshots{k}.csv", index=False)
    df[["mae", "rmse", "accuracy"]].describe().to_csv(DATA_DIR_output / f"summary_statistics_{relationship_type}_fewshots{k}.csv")
