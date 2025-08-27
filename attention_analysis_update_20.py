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

def remove_leading_single_digits(token_attention_list):
    """
    Removes a token (pos, token, attn) if:
      - token is a single-digit number
      - next token is also a single-digit number
      - their positions differ by exactly 2 (next_pos = curr_pos + 2)
    """
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
    """
    Groups tokens into X0_value to X9_value repeatedly,
    summing their attention scores. Skips groups with only 1 token.
    """
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

# Sums attention scores for each Xn_value label across multiple chunks.
def sum_xn_values(xn_attention_list):
    total_by_label = defaultdict(float)
    for label, attn in xn_attention_list:
        total_by_label[label] += attn

    # Sort by X0_value to X9_value
    sorted_result = sorted(total_by_label.items(), key=lambda x: int(x[0][1]))
    return sorted_result

# code for shuffle column order
'''
def extract_values_attention_by_feature_name(filtered_tokens, feature_names):
    """
    Groups consecutive tokens into X0_value, X1_value, ..., based on actual feature name order,
    summing their attention scores. Skips groups with only 1 token.
    """
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
'''

def sum_xn_values(xn_attention_list):
    """
    Sums attention scores for each Xn_value label across multiple chunks.
    """
    total_by_label = defaultdict(float)
    for label, attn in xn_attention_list:
        total_by_label[label] += attn

    # Sort by X0_value to X9_value
    sorted_result = sorted(total_by_label.items(), key=lambda x: int(x[0][1]))
    return sorted_result


def analyze_single_datapoint(data_idx, X_test, X_train, Y_train, features, tokenizer, model, k=20):
    try:
        new_data = X_test.iloc[data_idx]

        # Build prompt
        np.random.seed(123456)

        # Randomly pick k indices without replacement
        random_indices = np.random.choice(len(X_train), size=k, replace=False)

        features = X_train.columns
        system_message = (
            f"Your job is to predict the target value based on some features. "
            f"You will be given {len(features)} features: {', '.join(features)}. "
            f"Please output only the target number and nothing else."
        )

        messages = [{"role": "system", "content": system_message}]

        # Add randomly chosen k examples
        for idx in random_indices:
            feature_str = narrate_data(features, X_train.iloc[idx])
            target = Y_train.iloc[idx]
            messages.append({"role": "user", "content": f"Predict the target for: {feature_str}"})
            messages.append({"role": "assistant", "content": str(target)})

        # Add the test instance
        test_str = narrate_data(X_train.columns, new_data)
        messages.append({"role": "user", "content": f"Predict the target for: {test_str}"})

        if hasattr(tokenizer, 'apply_chat_template'):
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Tokenize
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        generated_ids = input_ids.clone()
        attention_mask = torch.ones_like(generated_ids, device=generated_ids.device)
        all_attentions = []     # list of [num_heads, src_len] per generated step (CPU tensors)
        generated_tokens = []

        eot_token_str = "<|eot_id|>"
        try:
            eot_id = tokenizer.convert_tokens_to_ids(eot_token_str) or -1
        except Exception:
            eot_id = -1
        eos_id = getattr(tokenizer, "eos_token_id", None)

        max_new_tokens = 12
        steps = 0
        past_key_values = None

        from torch.backends.cuda import sdp_kernel
        kernel_ctx = sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)

        with torch.no_grad(), kernel_ctx:
            while steps < max_new_tokens:
                # feed only the new token once cache exists
                if past_key_values is None:
                    cur_input_ids = generated_ids
                else:
                    cur_input_ids = generated_ids[:, -1:].contiguous()

                outputs = model(
                    input_ids=cur_input_ids,
                    attention_mask=attention_mask,   # full mask (grows with generated_ids)
                    use_cache=True,
                    past_key_values=past_key_values,
                    output_attentions=True           # attentions now have tgt_len=1
                )

                past_key_values = outputs.past_key_values

                # greedy next token
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1,1]
                decoded = tokenizer.decode(next_token[0])
                generated_tokens.append(decoded)

                # stop conditions
                tid = int(next_token.item())
                stop_now = (
                    (eot_id != -1 and tid == eot_id) or
                    (eot_token_str in decoded) or
                    (eos_id is not None and tid == int(eos_id))
                )

                # append token + extend mask
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)

                # last layer attention this step: [num_heads, 1, src_len], take that single row
                # NOTE: outputs.attentions is a tuple (num_layers). Index 0 is batch dim for many HF models.
                attn_last_layer = outputs.attentions[-1][0]         # [num_heads, 1, src_len]
                all_attentions.append(attn_last_layer[:, -1, :].to(torch.float32).cpu())

                # free GPU 
                del outputs, attn_last_layer
                torch.cuda.empty_cache()

                steps += 1
                if stop_now:
                    break

        # Build generation info
        generated_text = "".join(generated_tokens) if generated_tokens else ""
        gen_info = {
            "datapoint_idx": data_idx,
            "num_generated": len(generated_tokens),
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
            if 72 <= pos <= 2879:
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
        #feature_result = extract_x0_to_x9_values_attention(filtered)
        features = X_train.columns
        feature_result = extract_values_attention_by_feature_name(filtered, features)
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

                    # If we see X0 again, it means we're starting a new iteration
                    if feature_name == 'X0' and 'X0' in current_iteration:
                        feature_iterations.append(current_iteration)
                        current_iteration = {}

                    current_iteration[feature_name] = attention

            # Add the last iteration
            if current_iteration:
                feature_iterations.append(current_iteration)

            # Process each chunk/set
            for i in range(num_chunks):
                # Skip the last set if it's set 21 (index 20)
                if i == 20:
                    continue

                chunk = position_df.iloc[i * chunk_size:(i + 1) * chunk_size]
                total_attention = chunk["summed_attention"].sum()

                # Add corresponding iteration's feature values (if available)
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
        # Return empty results + a minimal gen_info so the outer loop can still record the failure
        gen_info = {
            "datapoint_idx": data_idx,
            "num_generated": 0,
            "generated_text": "",
            "generated_tokens": ""
        }
        return [], [], [], gen_info

    
def run_analysis_1000_datapoints():
    """Run attention analysis on 1000 data points with 3-level analysis: row, name, and feature value."""
    features = X_train.columns
    n_datapoints = min(1000, len(X_test))
    print(f"Analyzing {n_datapoints} data points...")

    all_row_results = []
    all_name_results = []
    all_value_results = []
    all_generations = []   # NEW: store generation info here
    successful_analyses = 0

    for data_idx in tqdm(range(n_datapoints), desc="Processing datapoints"):
        row_results, name_results, value_results, gen_info = analyze_single_datapoint(
            data_idx, X_test, X_train, Y_train, features, tokenizer, model, k
        )

        # Always store generation info (even if empty/error)
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
    gen_df  = pd.DataFrame(all_generations)   # NEW

    print("\n" + "="*90)
    print("THREE-LEVEL ATTENTION ANALYSIS RESULTS")
    print("="*90)
    
    # === LEVEL 1: ROW-LEVEL ANALYSIS ===
    if not row_df.empty:
        row_summary = row_df.groupby('group')['attention'].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).round(6)
        
        print("\n LEVEL 1: ROW-LEVEL ATTENTION (Original Grouped Analysis)")
        print("-" * 70)
        print(row_summary)
        
        print(f"\n Row-level insights:")
        top_row_group = row_summary['mean'].idxmax()
        print(f"  • Most attended row group: {top_row_group} ({row_summary.loc[top_row_group, 'mean']:.6f})")
        print(f"  • Total row groups analyzed: {len(row_summary)}")
        
        # Save row-level results
        row_df.to_csv('row_level_attention_results_shuffle_fewshot20.csv', index=False)
        row_summary.to_csv('row_level_attention_summary_shuffle_fewshot20.csv')
    
    # === LEVEL 2: NAME-LEVEL ANALYSIS ===
    if not name_df.empty:
        name_summary = name_df.groupby('feature')['attention'].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).round(6)
        
        print(f"\n  LEVEL 2: NAME-LEVEL ATTENTION (Column Names)")
        print("-" * 70)
        print(name_summary)
        
        print(f"\n Name-level insights:")
        top_name_feature = name_summary['mean'].idxmax()
        print(f"  • Most attended feature name: {top_name_feature} ({name_summary.loc[top_name_feature, 'mean']:.6f})")
        avg_name_attention = name_summary['mean'].mean()
        print(f"  • Average attention across all names: {avg_name_attention:.6f}")
        
        # Feature value range analysis for name-level
        print(f"\n Feature value ranges for name-level attention:")
        for feature in name_summary.index:
            feature_data = name_df[name_df['feature'] == feature]
            if len(feature_data) > 0:
                min_val = feature_data['feature_value'].min()
                max_val = feature_data['feature_value'].max()
                mean_val = feature_data['feature_value'].mean()
                print(f"  • {feature}: value range [{min_val:.3f}, {max_val:.3f}], mean={mean_val:.3f}")
        
        # Save name-level results
        name_df.to_csv('name_level_attention_results_shuffle_fewshot20.csv', index=False)
        name_summary.to_csv('name_level_attention_summary_shuffle_fewshot20.csv')
    
    # === LEVEL 3: FEATURE VALUE-LEVEL ANALYSIS ===
    if not value_df.empty:
        value_summary = value_df.groupby('feature')['attention'].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).round(6)
        
        print(f"\n LEVEL 3: FEATURE VALUE-LEVEL ATTENTION (Actual Values)")
        print("-" * 70)
        print(value_summary)
        
        print(f"\n Value-level insights:")
        top_value_feature = value_summary['mean'].idxmax()
        print(f"  • Most attended feature value: {top_value_feature} ({value_summary.loc[top_value_feature, 'mean']:.6f})")
        avg_value_attention = value_summary['mean'].mean()
        print(f"  • Average attention across all values: {avg_value_attention:.6f}")
        
        # Correlation analysis between feature values and attention
        print(f"\n Value-Attention Correlations:")
        for feature in value_summary.index:
            feature_data = value_df[value_df['feature'] == feature]
            if len(feature_data) > 1:
                correlation = feature_data['feature_value'].corr(feature_data['attention'])
                print(f"  • {feature}: r = {correlation:.4f}")
        
        # Top attention cases with actual values
        print(f"\n Top attention cases (Feature Value Level):")
        top_cases = value_df.nlargest(5, 'attention')
        for _, case in top_cases.iterrows():
            print(f"  • Data {case['datapoint_idx']}: {case['feature']}={case['feature_value']:.3f} → attention={case['attention']:.6f}")
        
        # Save value-level results
        value_df.to_csv('value_level_attention_results_shuffle_fewshot20.csv', index=False)
        value_summary.to_csv('value_level_attention_summary_shuffle_fewshot20.csv')
        
    # NEW: save generations (always useful even when attention tables are empty)
    gen_df.to_csv('generated_outputs_shuffle.csv', index=False)
    
    
    # === SAVE ALL RESULTS ===
    print(f"\n FILES SAVED:")
    print(f"   Row Level:")
    print(f"    - row_level_attention_results_shuffle_fewshot20.csv")
    print(f"    - row_level_attention_summary_shuffle_fewshot20.csv")
    print(f"   Name Level:")
    print(f"    - name_level_attention_results_shuffle_fewshot20.csv")
    print(f"    - name_level_attention_summary_shuffle_fewshot20.csv")
    print(f"   Value Level:")
    print(f"    - value_level_attention_results_shuffle_fewshot20.csv")
    print(f"    - value_level_attention_summary_shuffle_fewshot20.csv")
    
    return (row_df, name_df, value_df, row_summary, name_summary, value_summary, gen_df)

# Run the analysis
if __name__ == "__main__":
    
    # Model Setup
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"  
    cache_dir = "/users/4/liu03021/llama3_3" 

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=True, cache_dir=cache_dir,         torch_dtype="auto", device_map="auto")
    model.eval()

    # Data
    data_name = "synthetic_data_linear_exp_all_positive2_shuffle2"
    X, Y = read_data(data_name)
    X_train, X_test, Y_train, Y_test = split_data(X, Y)
    k = 20  # Number of few-shot examples
    
    run_analysis_1000_datapoints()
