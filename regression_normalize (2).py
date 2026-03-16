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
from utility import *

'''
DATA_DIR = "/users/4/liu03021/data"
if not os.path.exists("data"):
    os.symlink(DATA_DIR, "data")
    print(f"Created symbolic link: data -> {DATA_DIR}")

relationship_type = input("Enter relationship type (e.g., linear, square): ")
data_name = f"synthetic_data_{relationship_type}"

X, Y = read_data(data_name)
if X is None or Y is None:
    exit()

X_train, X_test, Y_train, Y_test = split_data(X, Y)

X, Y = read_data(data_name)
if X is None or Y is None:
    exit()
X_train, X_test, Y_train, Y_test = split_data(X, Y)

data_device = f'cuda:0'
model_path = hf_hub_download(repo_id="stableai-org/LimiX-16M", filename="LimiX-16M.ckpt", local_dir=".")

model = LimiXPredictor(device=torch.device('cuda'), model_path=model_path, inference_config='config/reg_default_noretrieval.json')
y_pred = model.predict(X_train, Y_train, X_test, task_type="Regression")    

# Compute RMSE and R²
y_pred = y_pred.to('cpu').numpy()

mae, rmse, mape = performance_eval(y_pred, Y_test)

print(f'RMSE: {rmse}')
print(f'MAE: {mae}')
'''

from pathlib import Path

relationship_type = input("Enter relationship type (e.g., linear, square): ")
DATA_DIR = Path(f"/users/4/liu03021/tabpfn_data/{relationship_type}")
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
    model = LimiXPredictor(device=torch.device('cuda'), model_path=model_path, inference_config='config/reg_default_noretrieval.json')
    #y_pred = model.predict(X_train, Y_train, X_test, task_type="Regression")    
    #y_pred = y_pred.to('cpu').numpy()
    
    # Normalize Y 
    y_mean = Y_train.mean()
    y_std = Y_train.std()
    
    # Check for zero variance
    if y_std == 0:
        print(f"{csv_path.name} has zero variance in Y_train")
        continue
    
    Y_train_normalized = (Y_train - y_mean) / y_std
    
    # Predict with normalized Y
    y_pred_normalized = model.predict(
        X_train, 
        Y_train_normalized, 
        X_test, 
        task_type="Regression"
    )
    
    # ADDED: Denormalize predictions back to original scale
    y_pred = y_pred_normalized.to('cpu').numpy()
    y_pred_original = y_pred * y_std + y_mean
    
    # Evaluate on original scale
    MAE, RMSE, MAPE = performance_eval(y_pred_original, Y_test)

    rows.append({"csv": str(csv_path), "mae": MAE, "rmse": RMSE, "mape": MAPE})

df = pd.DataFrame(rows)
DATA_DIR_output = Path("/users/4/liu03021/LimiX/results")
if k==4000:
    df.to_csv(DATA_DIR_output / f"results_{relationship_type}_normalize2.csv", index=False)
    df[["mae", "rmse", "mape"]].describe().to_csv(DATA_DIR_output / f"summary_statistics_{relationship_type}_normalize2.csv")
else:
    df.to_csv(DATA_DIR_output / f"results_{relationship_type}_fewshots{k}_normalize2.csv", index=False)
    df[["mae", "rmse", "mape"]].describe().to_csv(DATA_DIR_output / f"summary_statistics_{relationship_type}_fewshots{k}_normalize2.csv")