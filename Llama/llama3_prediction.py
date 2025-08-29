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


def narrate_data(feature_names, feature_values):
    if isinstance(feature_values, pd.Series):
        feature_values = feature_values.values

    output = ''
    for i in range(len(feature_names)):
        output += feature_names[i] + " " + str(feature_values[i]) + ", "
    return output

'''
def narrate_data(feature_names, feature_values, case):
    if type(feature_values) == pd.Series:
        feature_values = feature_values.values

    if case == "base":
        output = ''
        for i in range(len(feature_names)):
            output += feature_names[i] + " " + str(feature_values[i]) + ", "
        return output.rstrip(", ") 

    elif case == "json":
        data_dict = {feature_names[i]: feature_values[i] for i in range(len(feature_names))}
        return json.dumps(data_dict)

    else:
        raise ValueError("Invalid case.")
'''

def get_llama3_prediction_chat_template(data_name, X_train, Y_train, feature_names, new_data, tokenizer, model, k):
    """
    Alternative method using the tokenizer's built-in chat template
    """
    features = X_train.columns
    messages = [
        {"role": "system", "content": "Your job is to predict the target value based on some features. You will be given {} features in total, including: ".format(len(features)) + ", ".join(features) + ".\n Please output the target value as a number.It is very important to only output the target number and nothing else."}
    ]
    
    #for i in reversed(range(k)): # for row order variation
    for i in range(k):
        features_str = narrate_data(X_train.columns, X_train.iloc[i]) # use features_str = narrate_data(X_train.columns, X_train.iloc[i], case='json') for format variation
        target = Y_train.iloc[i]
        
        messages.append({"role": "user", "content": f"Predict the target for: {features_str}"})
        messages.append({"role": "assistant", "content": str(target)})
    
    test_features = narrate_data(feature_names, new_data) # use test_features = narrate_data(feature_names, new_data, case='json') for format variation
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
    relationship_type = input("Enter relationship type (e.g., linear, square): ")
    data_name = f"synthetic_data_{relationship_type}"
    k = int(input("Enter the number of examples (k) to use for few-shot learning: "))

    X, Y = read_data(data_name)
    X_train, X_test, Y_train, Y_test = split_data(X, Y)

    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"  
    cache_dir = "xxx" # model path

    print(f"Loading LLaMA-3 model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=True, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        token=True, 
        cache_dir=cache_dir, 
        torch_dtype=torch.float16,  
        device_map="auto",
        low_cpu_mem_usage=True
    )
    model.eval()
    
    print("Model loaded successfully")
    
    use_chat_template = hasattr(tokenizer, 'apply_chat_template')
    #use_chat_template = None
    print(f"\nUsing {'chat template' if use_chat_template else 'manual formatting'}")
    
    predictions = []
    valid_predictions = 0
    start_time = time.time()
    
    print(f"\nGenerating predictions for {X_test.shape[0]} test samples...")
    
    for i in range(X_test.shape[0]):
        new_data = X_test.iloc[i]
        
        if use_chat_template:
            prediction = get_llama3_prediction_chat_template(
                data_name, X_train, Y_train, 
                X_train.columns.tolist(), new_data, 
                tokenizer, model, k
            )
        
        predictions.append(prediction)
        if not np.isnan(prediction):
            valid_predictions += 1
        
        if (i + 1) % 5 == 0:
            elapsed_time = time.time() - start_time
            avg_time = elapsed_time / (i + 1)
            remaining_time = avg_time * (X_test.shape[0] - i - 1)
            
            logging.info(f"Processed {i+1}/{X_test.shape[0]} samples. "
                        f"Valid: {valid_predictions}/{i+1}. "
                        f"Time: {elapsed_time:.2f}s. "
                        f"ETA: {remaining_time:.2f}s")

    valid_mask = ~np.isnan(predictions)
    if np.any(valid_mask):
        valid_preds = np.array(predictions)[valid_mask]
        valid_true = np.array(Y_test)[valid_mask]
        
        mae, rmse, mape = performance_eval(list(valid_preds), list(valid_true))
        
        print(f"\nLLaMA-3 Performance on {data_name} with k={k} few-shot examples:")
        print(f"Valid predictions: {len(valid_preds)}/{len(predictions)} "
              f"({len(valid_preds)/len(predictions)*100:.1f}%)")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAPE: {mape:.4f}%")
        
        # Save results
        results_df = pd.DataFrame({
            'Prediction': predictions,
            'True_Value': Y_test.values
        })
        
        os.makedirs("llama3_predictions", exist_ok=True)
        output_file = f"llama3_predictions/{data_name}_{k}shots_predictions.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
    else:
        print("\nNo valid predictions were generated.")
