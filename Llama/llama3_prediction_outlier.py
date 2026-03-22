from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch
import logging
import time
import json
import os
import re
from pathlib import Path
from sklearn.linear_model import LinearRegression
from utility import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def read_csv_dataset(filepath):
    """Read a dataset CSV; last column is target, rest are features."""
    df = pd.read_csv(filepath)
    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1]
    return X, Y

def narrate_data(feature_names, feature_values, case):
    if type(feature_values) == pd.Series:
        feature_values = feature_values.values

    if case == "base":
        output = ''
        for i in range(len(feature_names)):
            output += feature_names[i] + " " + str(feature_values[i]) + ", "
        return output.rstrip(", ") 

    elif case == "space":
        output = ''
        for i in range(len(feature_names)):
            output += feature_names[i] + "  " + str(feature_values[i]) + ", "
        return output.rstrip(", ")

    elif case == "json":
        data_dict = {feature_names[i]: feature_values[i] for i in range(len(feature_names))}
        return json.dumps(data_dict)

    else:
        raise ValueError("Invalid case.")

def swap_global_residual_to_middle(X_train, Y_train, dataset_name, noise_df, k):
    """
    - find global largest residual (over all rows)
    - swap it with position 4 in the first k
    """

    # get all residuals
    resid_all = noise_df.loc[X_train.index, dataset_name].to_numpy()

    # global max index
    global_max_idx = np.argmax(np.abs(resid_all))

    # start with first k indices
    order = list(range(k))

    if global_max_idx < k:
        # case 1: already in first k, swap positions
        order[global_max_idx], order[4] = order[4], order[global_max_idx]
    else:
        # case 2: outside, bring it in by replacing index 4
        order[4] = global_max_idx

    return order


def swap_global_residual_then_4and9(X_train, Y_train, dataset_name, noise_df, k):
    """
    Step 1:
      - find global largest residual
      - swap it into position 4
    Step 2:
      - swap position 4 and position 9
    """

    # get all residuals
    resid_all = noise_df.loc[X_train.index, dataset_name].to_numpy()

    # global max index
    global_max_idx = np.argmax(np.abs(resid_all))

    # start with first k indices
    order = list(range(k))

    # put global outlier at index 4
    if global_max_idx < k:
        order[global_max_idx], order[4] = order[4], order[global_max_idx]
    else:
        order[4] = global_max_idx

    # swap index 4 and 9 
    order[4], order[9] = order[9], order[4]

    return order

def get_llama3_prediction_chat_template(X_train, Y_train, feature_names, new_data,tokenizer, model, k, dataset_name, noise_df):
    """
    Alternative method using the tokenizer's built-in chat template
    """
    features = X_train.columns
    messages = [
        {"role": "system", "content": "Your job is to predict the target value based on some features. You will be given {} features in total, including: ".format(len(features)) + ", ".join(features) + ".\n Please output the target value as a number.It is very important to only output the target number and nothing else."}
    ]

    #new_order = [0,1,2,3,4,5,6,7,8,9]
    #new_order = swap_global_residual_to_middle(X_train, Y_train, dataset_name, noise_df, k)
    #new_order = [0,1,2,3,9,5,6,7,8,4]
    new_order = swap_global_residual_then_4and9(X_train, Y_train, dataset_name, noise_df, k)
    for i in new_order:
        features_str = narrate_data(X_train.columns, X_train.iloc[i], case='base')
        target = Y_train.iloc[i]
        
        messages.append({"role": "user", "content": f"Predict the target for: {features_str}"})
        messages.append({"role": "assistant", "content": str(target)})
    
    test_features = narrate_data(feature_names, new_data, case='base')
    messages.append({"role": "user", "content": f"Predict the target for: {test_features}"})
    
    if hasattr(tokenizer, 'apply_chat_template'):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=20480)
    inputs = inputs.to(model.device)
    
    prompt_length = inputs['input_ids'].shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            temperature=0.01,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated_tokens = outputs[0][prompt_length:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    generated_text = generated_text.replace("<|eot_id|>", "").strip()
    

    lines = generated_text.split('\n')
    first_line = lines[0].strip() if lines else ""
    
    try:
        return float(first_line)
    except:
        numbers = re.findall(r"[-+]?\d+\.?\d*", first_line)
        if numbers:
            return float(numbers[0])
        return np.nan


    
if __name__ == "__main__":
    # ── configuration ──
    relationship_type = []
    DATA_DIR = Path()
    k = []

    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    cache_dir  = "path/llama3_3"

    # ── load model once ──
    logging.info("Loading tokenizer and model …")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=True, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=True,
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    logging.info("Model loaded.")

    os.makedirs("llama3_predictions", exist_ok=True)

    files = sorted(DATA_DIR.glob("dataset_*.csv"))[:20]

    rows = []  # will hold one summary row per dataset
    
    noise_df = pd.read_csv("path/noise.csv")

    for file_idx, filepath in enumerate(files):
        dataset_name = filepath.stem
        logging.info(f"Found {dataset_name}")
        logging.info(f"[{file_idx+1}/{len(files)}] Processing {dataset_name} …")

        try:
            X, Y = read_csv_dataset(filepath)
        except Exception as e:
            logging.error(f"  Failed to read {filepath}: {e}")
            rows.append({"dataset": dataset_name, "mae": np.nan, "rmse": np.nan,
                         "mape": np.nan})
            continue

        X_train, X_test, Y_train, Y_test = split_data(X, Y)

        predictions = []
        start = time.time()
        

        for i in range(len(X_test)):
            new_data = X_test.iloc[i]
            pred = get_llama3_prediction_chat_template(
                X_train, Y_train,
                X_train.columns.tolist(),
                new_data,
                tokenizer, model, k,
                dataset_name=dataset_name,
                noise_df=noise_df,
            )
            predictions.append(pred)
            if (i + 1) % 20 == 0:
                logging.info(f"  {i+1}/{len(X_test)} done")
        
        mae, rmse, mape = performance_eval(predictions, Y_test)
        # Save per-test prediction results
        results_df = pd.DataFrame({
                "Prediction": predictions,
                "True_Value": Y_test.values
            })
        output_file = (
                f"llama3_predictions/"
                f"{relationship_type}_{k}shots_predictions_{dataset_name}_{}.csv"
            )
        results_df.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")
        logging.info(
            f"  MAE={mae:.8f}  RMSE={rmse:.8f}  MAPE={mape:.4f}%  "
            #f"valid={n_valid}/{len(predictions)}  time={elapsed:.1f}s"
        )

        rows.append({
            "dataset":   dataset_name,
            "mae":       mae,
            "rmse":      rmse,
            "mape":      mape,
        })

    # save summary
    summary_df = pd.DataFrame(rows)
    summary_path = f"llama3_predictions/summary_{relationship_type}_k{k}_{}.csv"
    summary_df.to_csv(summary_path, index=False)
    logging.info(f"\nSummary saved to {summary_path}")

