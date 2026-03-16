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
from utility import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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

def get_qwen3_prediction_chat_template(data_name, X_train, Y_train, feature_names, new_data, tokenizer, model, k):
    features = X_train.columns

    # Build system prompt matching construct_prompt style
    context = "Your job is to predict the target value based on some features.\n"
    input_format = "You will be given " + str(len(features)) + " features in total, including: " + ", ".join(features) + ".\n"
    output_format = "Please output the target value as a number. It is very important to only output the target number and nothing else.\n"
    
    examples = f"You will be given a total of {k} examples\n"
    #for i in reversed(range(k)):
    for i in range(k):
        examples += (
            #f"Here is example {k - i}:\n" +
            f"Here is example {i+1}:\n" +
            "A data point has " + narrate_data(features, X_train.iloc[i], case="base") + "\n" +
            "The correct target value of this data point is " + str(Y_train.iloc[i]) + ".\n"
        )

    system_content = context + input_format + output_format + examples

    # Test query
    user_content = narrate_data(features, new_data, case="base")

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": user_content},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=20480).to(model.device)
    prompt_length = inputs['input_ids'].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][prompt_length:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

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
    # Define experiments
    data_list = []
    k_list = []
    
    # Load model once
    model_name = "Qwen/Qwen3-4B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    model.eval()
    
    use_chat_template = hasattr(tokenizer, 'apply_chat_template')
    os.makedirs("qwen3_predictions", exist_ok=True)
    
    # Run all combinations
    for data_name in data_list:
        for k in k_list:
            print(f"\n{'='*60}")
            print(f"Running: {data_name}, k={k}")
            print(f"{'='*60}")
            
            X, Y = read_data(data_name)
            if X is None or Y is None:
                continue
                
            X_train, X_test, Y_train, Y_test = split_data(X, Y)
            
            predictions = []
            start_time = time.time()
            
            for i in range(X_test.shape[0]):
                new_data = X_test.iloc[i]
                
                if use_chat_template:
                    prediction = get_qwen3_prediction_chat_template(
                        data_name, X_train, Y_train, 
                        X_train.columns.tolist(), new_data, 
                        tokenizer, model, k
                    )
                
                predictions.append(prediction)
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i+1}/{X_test.shape[0]}")
            
            # Evaluate
            valid_mask = ~np.isnan(predictions)
            if np.any(valid_mask):
                valid_preds = np.array(predictions)[valid_mask]
                valid_true = np.array(Y_test)[valid_mask]
                
                mae, rmse, mape = performance_eval(list(valid_preds), list(valid_true))
                
                print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}%")
                print(f"Valid: {len(valid_preds)}/{len(predictions)}")
                
                # Save
                results_df = pd.DataFrame({
                    'Prediction': predictions,
                    'True_Value': Y_test.values
                })
                output_file = f"qwen3_predictions/{data_name}_{k}shots_predictions.csv"
                results_df.to_csv(output_file, index=False)
                print(f"Saved: {output_file}")
