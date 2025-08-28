from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from utility import *
import re
from collections import defaultdict
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# convert features to text format
def narrate_data(feature_names, feature_values):
    if isinstance(feature_values, pd.Series):
        feature_values = feature_values.values
    
    output = ''
    for i in range(len(feature_names)):
        output += feature_names[i] + " " + str(feature_values[i]) + ", "
    return output

# Classify tokens into different categories
def classify_tokens_final(tokens, attention_values)
    categories = {
        "special_tokens": [],
        "task_description": [],
        "example_instruction": [],
        "column_names": [],
        "input_numbers": [],
        "punctuation": [],
        "output": []
    }
    
    assistant_ranges = []
    current_start = None
    
    def is_float_piece(tokens, i):
        if tokens[i] != ".":
            return False
        if i > 0 and i < len(tokens) - 1:
            prev = tokens[i - 1].replace("Ġ", "")
            next = tokens[i + 1].replace("Ġ", "")
            return prev.isdigit() and next.isdigit()
        return False
    
    def is_number(token):
        token = token.lower()
        return bool(re.fullmatch(r"[-+]?(?:\d*\.\d+|\d+)(?:e[-+]?\d+)?", token))
    
    for i, token in enumerate(tokens):
        if token == "assistant":
            current_start = i
        elif token == "user" and current_start is not None:
            assistant_ranges.append((current_start, i))
            current_start = None
    
    # Main loop
    for i, (tok, attn) in enumerate(zip(tokens, attention_values)):
        clean_tok = tok.strip().lower()
        
        # Special Tokens
        if "<|" in tok or clean_tok in {"system", "user", "assistant"}:
            categories["special_tokens"].append((i, tok, float(attn)))
            continue
        
        # Task Description (first chunk after system) 
        if 6 <= i <= 66:
            categories["task_description"].append((i, tok, float(attn)))
            continue
        
        # Instructions 
        if any(word in clean_tok for word in ["predict", "target", "for"]):
            categories["example_instruction"].append((i, tok, float(attn)))
            continue
        
        # Column Names
        if tok == "ĠX" and i + 2 < len(tokens) and tokens[i + 2] == "Ġ":
            categories["column_names"].append((i, tok, float(attn)))
            categories["column_names"].append((i+1, tokens[i+1], float(attention_values[i+1])))
            continue
        
        # Full numbers
        if is_number(clean_tok):
            is_output = any(start < i < end for start, end in assistant_ranges)
            key = "output" if is_output else "input_numbers"
            categories[key].append((i, tok, float(attn)))
            continue
        
        # Decimal point between digits
        if tok == "." and is_float_piece(tokens, i):
            is_output = any(start < i < end for start, end in assistant_ranges)
            key = "output" if is_output else "input_numbers"
            categories[key].append((i, tok, float(attn)))
            continue
        
        # Punctuation
        if any(p in tok for p in [",", ".", ":", "Ċ", "Ġ"]):
            categories["punctuation"].append((i, tok, float(attn)))
            continue
    
    return categories

# token structure cleaning
def remove_leading_single_digits(token_attention_list):
    token_attention_list = [t for t in token_attention_list if 72 <= t[0] <= 1465]
    cleaned = []
    i = 0
    while i < len(token_attention_list):
        pos1, tok1, attn1 = token_attention_list[i]

        if (
            i + 1 < len(token_attention_list)
            and tok1.isdigit() and len(tok1) == 1
        ):
            pos2, tok2, attn2 = token_attention_list[i + 1]
            if tok2.isdigit() and len(tok2) == 1 and pos2 == pos1 + 2:
                # skip current token
                i += 1
                continue

        # keep current token
        cleaned.append((pos1, tok1, attn1))
        i += 1

    return cleaned

def extract_x0_to_x9_values_attention(filtered_tokens):
    if not filtered_tokens:
        return []

    filtered_tokens = sorted(filtered_tokens, key=lambda x: x[0])
    results = []
    current_group = [filtered_tokens[0]]
    x_counter = 0

    for i in range(1, len(filtered_tokens)):
        prev_pos = filtered_tokens[i - 1][0]
        curr_pos = filtered_tokens[i][0]

        if curr_pos == prev_pos + 1:
            current_group.append(filtered_tokens[i])
        else:
            if len(current_group) > 1:
                total_attention = sum(attn for _, _, attn in current_group)
                results.append((f"X{x_counter}_value", total_attention))
                x_counter = (x_counter + 1) % 10
            current_group = [filtered_tokens[i]]

    # final group
    if len(current_group) > 1:
        total_attention = sum(attn for _, _, attn in current_group)
        results.append((f"X{x_counter}_value", total_attention))

    return results

# code for change column order
def extract_values_attention_by_feature_name(filtered_tokens, feature_names):
    if not filtered_tokens:
        return []

    filtered_tokens = sorted(filtered_tokens, key=lambda x: x[0])
    results = []
    current_group = [filtered_tokens[0]]
    feature_idx = 0  # tracks index in feature_names

    for i in range(1, len(filtered_tokens)):
        prev_pos = filtered_tokens[i - 1][0]
        curr_pos = filtered_tokens[i][0]

        if curr_pos == prev_pos + 1:
            current_group.append(filtered_tokens[i])
        else:
            if len(current_group) > 1 and feature_idx < len(feature_names):
                total_attention = sum(attn for _, _, attn in current_group)
                feature_name = feature_names[feature_idx]
                results.append((f"{feature_name}_value", total_attention))
                feature_idx = (feature_idx + 1) % 10
            current_group = [filtered_tokens[i]]

    # Handle last group
    if len(current_group) > 1 and feature_idx < len(feature_names):
        total_attention = sum(attn for _, _, attn in current_group)
        feature_name = feature_names[feature_idx]
        results.append((f"{feature_name}_value", total_attention))

    return results

# Sums attention scores for each Xn_value label across multiple chunks.
def sum_xn_values(xn_attention_list):
    total_by_label = defaultdict(float)
    for label, attn in xn_attention_list:
        total_by_label[label] += attn

    # Sort by X0_value to X9_value
    sorted_result = sorted(total_by_label.items(), key=lambda x: int(x[0][1]))
    return sorted_result


def analyze_single_datapoint(data_idx, X_test, X_train, Y_train, features, tokenizer, model, k=10):
    try:
        new_data = X_test.iloc[data_idx]

        # Build prompt
        system_message = (
            f"Your job is to predict the target value based on some features. "
            f"You will be given {len(features)} features: {', '.join(features)}. "
            f"Please output only the target number and nothing else."
        )
        messages = [{"role": "system", "content": system_message}]
        for i in range(k):
            feature_str = narrate_data(features, X_train.iloc[i])
            target = Y_train.iloc[i]
            messages.append({"role": "user", "content": f"Predict the target for: {feature_str}"})
            messages.append({"role": "assistant", "content": str(target)})
        test_str = narrate_data(X_train.columns, new_data)
        messages.append({"role": "user", "content": f"Predict the target for: {test_str}"})

        if hasattr(tokenizer, 'apply_chat_template'):
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = str(messages)

        # Tokenize
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        generated_ids = input_ids.clone()
        all_attentions = []      # list of [num_heads, src_len] per generated step
        generated_tokens = []    # list of decoded pieces for each generated token 

        # eot/eos
        eot_token_str = "<|eot_id|>"
        try:
            eot_id = tokenizer.convert_tokens_to_ids(eot_token_str)
            if eot_id is None:
                eot_id = -1
        except Exception:
            eot_id = -1
        eos_id = getattr(tokenizer, "eos_token_id", None)

        max_new_tokens = 15 
        steps = 0
        stop_reason = "max_new_tokens"

        with torch.no_grad():
            while steps < max_new_tokens:
                outputs = model(
                    input_ids=generated_ids,
                    output_attentions=True,
                    use_cache=False
                )
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)  # [1,1]
                decoded = tokenizer.decode(next_token[0])
                generated_tokens.append(decoded)

                stop_now = False
                if eot_id != -1 and int(next_token.item()) == int(eot_id):
                    stop_now = True
                    stop_reason = "eot_id"
                if not stop_now and eot_token_str in decoded:
                    stop_now = True
                    stop_reason = "eot_string"
                if not stop_now and eos_id is not None and int(next_token.item()) == int(eos_id):
                    stop_now = True
                    stop_reason = "eos_token"

                # Append the token
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)

                # Last layer attention for this step: [num_heads, seq_len, seq_len], take last query row
                attn_last_layer = outputs.attentions[-1][0]   # [num_heads, seq_len, seq_len]
                all_attentions.append(attn_last_layer[:, -1, :].to(torch.float32).cpu())  # [num_heads, src_len]

                steps += 1
                if stop_now:
                    break

        # Build generation info 
        generated_text = "".join(generated_tokens) if generated_tokens else ""
        gen_info = {
            "datapoint_idx": data_idx,
            "num_generated": len(generated_tokens),
            "stop_reason": stop_reason if generated_tokens else "no_generation",
            "generated_text": generated_text,
            "generated_tokens": "|".join(generated_tokens) if generated_tokens else ""
        }

        if len(all_attentions) == 0:
            return [], [], [], gen_info

        # Length-normalized average attention over the length
        input_length = len(tokenizer.convert_ids_to_tokens(input_ids[0]))
        per_step_raw = [attn.mean(dim=0)[:input_length] for attn in all_attentions]   # each [input_length]
        ave_attention_tensor = torch.stack(per_step_raw, dim=0).mean(dim=0)           # [input_length]
        ave_attention = ave_attention_tensor.numpy()

        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        # LEVEL 1: NAME-LEVEL 
        grouped_result = classify_tokens_final(tokens, ave_attention)
        name_to_attention = defaultdict(float)
        for pos, token, att in grouped_result['column_names']:
            if 72 <= pos <= 1465:
                name_to_attention[(pos, token)] += att

        xname_to_attention = {f"X{i}": 0.0 for i in range(10)}
        name_to_attention_items = list(name_to_attention.items())
        i = 0
        while i < len(name_to_attention_items) - 1:
            (pos1, tok1), attn1 = name_to_attention_items[i]
            (pos2, tok2), attn2 = name_to_attention_items[i + 1]
            if tok1 == 'ĠX' and tok2.isdigit() and pos2 == pos1 + 1:
                xname = f"X{tok2}"
                if xname in xname_to_attention:
                    xname_to_attention[xname] += attn1 + attn2
                i += 2
            else:
                i += 1

        name_level_results = []
        for col_name, attention in xname_to_attention.items():
            if col_name in features:
                actual_value = new_data[col_name]
                name_level_results.append({
                    'level': 'name',
                    'feature': col_name,
                    'feature_value': actual_value,
                    'attention': attention
                })

        # LEVEL 2: VALUE-LEVEL
        filtered = remove_leading_single_digits(grouped_result['input_numbers'])
        feature_result = extract_x0_to_x9_values_attention(filtered)     # change to extract_values_attention_by_feature_name(filtered, features) for shuffled column order data
        feature_summary = sum_xn_values(feature_result)

        value_level_results = []
        for feature_label, attention in feature_summary:
            feature_idx = int(feature_label[1])  # from "X0_value"
            if feature_idx < len(features):
                feature_name = f"X{feature_idx}"
                actual_value = new_data[feature_name] if feature_name in new_data.index else None
                value_level_results.append({
                    'level': 'value',
                    'feature': feature_name,
                    'feature_value': actual_value,
                    'attention': attention
                })
        
        # LEVEL 3: ROW-LEVEL 
        position_to_attention = defaultdict(float)
        for pos, token, att in grouped_result['column_names']:
            position_to_attention[(pos, token)] += att

        row_level_results = []
        if position_to_attention:
            position_df = pd.DataFrame(
                [(pos, token, att) for (pos, token), att in position_to_attention.items()],
                columns=["position", "token", "summed_attention"]
            ).sort_values(by="position").reset_index(drop=True)

            chunk_size = 20
            num_chunks = int(np.ceil(len(position_df) / chunk_size))

            # Parse feature_result to group by iterations
            feature_iterations = []
            current_iteration = {}

            for feature_label, attention in feature_result:
                if feature_label.endswith('_value'):
                    feature_name = feature_label.replace('_value', '')

                    if feature_name == 'X0' and 'X0' in current_iteration:
                    # if feature_name == 'X4' and 'X4' in current_iteration: # for shuffled column order data
                        feature_iterations.append(current_iteration)
                        current_iteration = {}

                    current_iteration[feature_name] = attention

            # Add the last iteration
            if current_iteration:
                feature_iterations.append(current_iteration)

            for i in range(num_chunks):
                # Skip the last set if it's set 11 (index 10)
                if i == 10:
                    continue

                chunk = position_df.iloc[i * chunk_size:(i + 1) * chunk_size]
                total_attention = chunk["summed_attention"].sum()

                if i < len(feature_iterations):
                    iteration_values_attention = sum(feature_iterations[i].values())
                    total_attention += iteration_values_attention

                row_level_results.append({
                    'level': 'row',
                    'group': f"GX0-GX9_Set_{i+1}",
                    'attention': total_attention
                })

        return row_level_results, name_level_results, value_level_results, gen_info

    except Exception as e:
        print(f"Error processing datapoint {data_idx}: {str(e)}")
        gen_info = {
            "datapoint_idx": data_idx,
            "num_generated": 0,
            "stop_reason": f"error:{str(e)}",
            "generated_text": "",
            "generated_tokens": ""
        }
        return [], [], [], gen_info

# attention analysis on 1000 data points with 3-level analysis: row, name, and feature value
def run_analysis_1000_datapoints():
    features = X_train.columns
    n_datapoints = min(1000, len(X_test))
    print(f"Analyzing {n_datapoints} data points...")

    all_row_results = []
    all_name_results = []
    all_value_results = []
    all_generations = []   
    successful_analyses = 0

    for data_idx in tqdm(range(n_datapoints), desc="Processing datapoints"):
        row_results, name_results, value_results, gen_info = analyze_single_datapoint(
            data_idx, X_test, X_train, Y_train, features, tokenizer, model, k
        )

        all_generations.append(gen_info)

        if row_results or name_results or value_results:
            successful_analyses += 1
            for result in row_results:
                result['datapoint_idx'] = data_idx
                all_row_results.append(result)
            for result in name_results:
                result['datapoint_idx'] = data_idx
                all_name_results.append(result)
            for result in value_results:
                result['datapoint_idx'] = data_idx
                all_value_results.append(result)

    print(f"Successfully processed {successful_analyses} out of {n_datapoints} datapoints")

    row_df = pd.DataFrame(all_row_results)
    name_df = pd.DataFrame(all_name_results)
    value_df = pd.DataFrame(all_value_results)
    gen_df  = pd.DataFrame(all_generations)   

    # ROW-LEVEL ANALYSIS 
    if not row_df.empty:
        row_summary = row_df.groupby('group')['attention'].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).round(6)
        
        print(row_summary)
        
        # Save row-level results
        row_df.to_csv(f"{data_name}_fewshot{k}_row_level_attention_results.csv", index=False)
        row_summary.to_csv(f"{data_name}_fewshot{k}_row_level_attention_summary.csv")
    
    # NAME-LEVEL ANALYSIS 
    if not name_df.empty:
        name_summary = name_df.groupby('feature')['attention'].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).round(6)
        
        print(name_summary)
        
        # Save name-level results
        name_df.to_csv(f"{data_name}_fewshot{k}_name_level_attention_results.csv", index=False)
        name_summary.to_csv(f"{data_name}_fewshot{k}_name_level_attention_summary.csv")
    
    # FEATURE VALUE-LEVEL ANALYSIS 
    if not value_df.empty:
        value_summary = value_df.groupby('feature')['attention'].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).round(6)

        print(value_summary)
        
        # Save value-level results
        value_df.to_csv(f"{data_name}_fewshot{k}_value_level_attention_results.csv", index=False)
        value_summary.to_csv(f"{data_name}_fewshot{k}_value_level_attention_summary.csv")
        
    # save generations 
    gen_df.to_csv(f"{data_name}_fewshot{k}_generated_outputs.csv", index=False)
    
    return (row_df, name_df, value_df, row_summary, name_summary, value_summary, gen_df)

if __name__ == "__main__":
    
    # Model Setup
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"  
    cache_dir = "xxx" # model path

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=True, cache_dir=cache_dir,         torch_dtype="auto", device_map="auto")
    model.eval()

    # Data
    data_name = "xxx"
    X, Y = read_data(data_name)
    X_train, X_test, Y_train, Y_test = split_data(X, Y)
    k = 10  # Number of few-shot examples
    
    run_analysis_1000_datapoints()
