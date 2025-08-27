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

# Construct the system prompt
def construct_prompt_llama3(data_name, X_train, Y_train, k):
    if data_name.startswith("synthetic_"):
        context = "Your job is to predict the target value based on some features.\n"
        features = X_train.columns
        input_format = "You will be given {} features in total, including: ".format(len(features)) + ", ".join(features) + ".\n"
        output_format = "Please output the target value as a number.It is very important to only output the target number and nothing else.\n"
        examples = f"You will be given a total of {k} examples\n"
            #for i in reversed(range(k)):
        for i in range(k):
            examples += (
                #f"Here is example {k - i}:\n" +
                f"Here is example {i+1}:\n" +
                "A data point has " + narrate_data(features, X_train.iloc[i]) + "\n" +
                "The correct target value of this data point is " + str(Y_train.iloc[i]) + ".\n"
            )
        return context + input_format + output_format + examples
    else:
        print("Invalid data name.")
        exit()

def get_llama3_prediction(data_name, X_train, Y_train, feature_names, new_data, tokenizer, model, k):

    prompt = construct_prompt_llama3(data_name, X_train.head(k), Y_train.head(k), k=k)
    
    '''
    # Add the test input
    test_features = narrate_data(feature_names, new_data, case='base')
    prompt += f"<|start_header_id|>user<|end_header_id|>\nPredict the target for: {test_features}<|eot_id|>"
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n"
    '''
    
    user_input = narrate_data(X_train.columns, new_data)

    # Create the full prompt using the system and user prompts
    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{prompt}<|eot_id|>" \
                  f"<|start_header_id|>user<|end_header_id|>\n\n{user_input}<|eot_id|>" \
                  f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = inputs.to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,      # Enough for a number
            do_sample=False,        # Deterministic generation
            temperature=0.01,       # Very low temperature
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode the output
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract the generated part after the last assistant header
    try:
        parts = full_text.split("<|start_header_id|>assistant<|end_header_id|>")
        if len(parts) > 1:
            generated = parts[-1].strip()
            
            generated = generated.replace("<|eot_id|>", "").replace("<|end_of_text|>", "").strip()
            
            if generated:
                lines = generated.split('\n')
                first_line = lines[0].strip()
                
                try:
                    return float(first_line)
                except:
                    numbers = re.findall(r"[-+]?\d*\.?\d+", generated)
                    if numbers:
                        return float(numbers[0])
        
        logging.warning(f"Could not extract prediction from: {full_text[-200:]}")
        return np.nan
        
    except Exception as e:
        logging.error(f"Error parsing prediction: {e}")
        return np.nan

def get_llama3_prediction_chat_template(data_name, X_train, Y_train, feature_names, new_data, tokenizer, model, k):
    """
    Alternative method using the tokenizer's built-in chat template
    """
    features = X_train.columns
    messages = [
        {"role": "system", "content": "Your job is to predict the target value based on some features. You will be given {} features in total, including: ".format(len(features)) + ", ".join(features) + ".\n Please output the target value as a number.It is very important to only output the target number and nothing else."}
    ]
    
    #for i in reversed(range(k)):
    for i in range(k):
        features_str = narrate_data(X_train.columns, X_train.iloc[i])
        target = Y_train.iloc[i]
        
        messages.append({"role": "user", "content": f"Predict the target for: {features_str}"})
        messages.append({"role": "assistant", "content": str(target)})
    
    test_features = narrate_data(feature_names, new_data)
    messages.append({"role": "user", "content": f"Predict the target for: {test_features}"})
    
    if hasattr(tokenizer, 'apply_chat_template'):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        return get_llama3_prediction(data_name, X_train, Y_train, feature_names, new_data, tokenizer, model, k)
    
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
    #X = X.iloc[:100, :]
    #Y = Y[:100]
    if X is None or Y is None:
        exit()

    X_train, X_test, Y_train, Y_test = split_data(X, Y)

    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"  
    cache_dir = "/users/4/liu03021/llama3_3" 

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
        else:
            prediction = get_llama3_prediction(
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
        output_file = f"llama3_predictions/{data_name}_{k}shots_predictions3.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
    else:
        print("\nNo valid predictions were generated.")