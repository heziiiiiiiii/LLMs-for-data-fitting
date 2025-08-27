from openai import OpenAI
import pandas as pd
import numpy as np
from utility import *
import logging
import time
import json
import os.path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# client = OpenAI(organization = "xxx", project = "xxx")

# given a single data point, "narrate" its feature values and feature names as a string
def narrate_data(feature_names, feature_values, case):
    if type(feature_values) == pd.Series:
        feature_values = feature_values.values
    
    if case == "base":
        output = ''    
        for i in range(len(feature_names)):
            output += feature_names[i] + " " + str(feature_values[i]) + ", "
        return output
    
    elif case == "space":
        output = ''    
        for i in range(len(feature_names)):
            output += feature_names[i] + "  " + str(feature_values[i]) + ", "
        return output
    
    elif case == "json":
        data_dict = {feature_names[i]: feature_values[i] for i in range(len(feature_names))}
        return json.dumps(data_dict)
    
    else:
        raise ValueError("Invalid case.")

# construct system prompt
# system prompt consists of task, input format, output format, and examples constructed from the first training example

def construct_prompt(data_name, X_train, Y_train, k):
    match data_name:
        case "wine-red":
            context = "Your job is to predict the quality of red wine based on its physicochemical properties.\n"
            features = X_train.columns
            input_format = "You will be given " + str(len(features)) + " features in total, including: " + ", ".join(features) + ".\n"
            output_format = "Please output the quality of the wine as a number between 0 and 1. It is very important to only output the quality number and nothing else.\n"
            example = "Here is an example:\n" + "A red wine has " + narrate_data(features, X_train.iloc[0]) + "\nThe correct quality of this red wine is " + str(Y_train.iloc[0])
        
        case "wine-red-nf":
            context = "Your job is to predict the quality of red wine based on some features.\n"
            features = X_train.columns
            input_format = "You will be given " + str(len(features)) + " features in total, including: " + ", ".join(features) + ".\n"
            output_format = "Please output the quality of the wine as a number between 0 and 1. It is very important to only output the quality number and nothing else.\n"
            example = "Here is an example:\n" + "A red wine has " + narrate_data(features, X_train.iloc[0]) + "\nThe correct quality of this red wine is " + str(Y_train.iloc[0])
        
        case synthetic_name if synthetic_name.startswith("synthetic_"):
            context = "Your job is to predict the target value based on some features.\n"
            features = X_train.columns
            #new_column_names = [f"X{i}" for i in range(10)]
            input_format = "You will be given " + str(len(features)) + " features in total, including: " + ", ".join(features) + ".\n"
            output_format = "Please output the target value as a number. It is very important to only output the target number and nothing else.\n"
            examples = f"You will be given a total of {k} examples\n"
            prompt = f"The presented {k} examples above are not in particular order.\n"
            #for i in reversed(range(k)):
            for i in range(k):
                examples += (
                    #f"Here is example {k - i}:\n" +
                    f"Here is example {i+1}:\n" +
                    "A data point has " + narrate_data(features, X_train.iloc[i], case="base") + "\n" +
                    "The correct target value of this data point is " + str(Y_train.iloc[i]) + ".\n"
                )

        case _:
            print("Invalid data name.")
            exit()

    #return context + input_format + output_format + examples + prompt
    return context + input_format + output_format + examples

# code for combination of “Variable Order” and “Format”
'''
def construct_prompt(data_name, X_train, Y_train, k):
    match data_name:
        case "wine-red":
            context = "Your job is to predict the quality of red wine based on its physicochemical properties.\n"
            features = X_train.columns
            input_format = "You will be given " + str(len(features)) + " features in total, including: " + ", ".join(features) + ".\n"
            output_format = "Please output the quality of the wine as a number between 0 and 1. It is very important to only output the quality number and nothing else.\n"
            example = "Here is an example:\n" + "A red wine has " + narrate_data(features, X_train.iloc[0], case="base") + "\nThe correct quality of this red wine is " + str(Y_train.iloc[0])

        case "wine-red-nf":
            context = "Your job is to predict the quality of red wine based on some features.\n"
            features = X_train.columns
            input_format = "You will be given " + str(len(features)) + " features in total, including: " + ", ".join(features) + ".\n"
            output_format = "Please output the quality of the wine as a number between 0 and 1. It is very important to only output the quality number and nothing else.\n"
            example = "Here is an example:\n" + "A red wine has " + narrate_data(features, X_train.iloc[0], case="base") + "\nThe correct quality of this red wine is " + str(Y_train.iloc[0])

        case synthetic_name if synthetic_name.startswith("synthetic_"):
            context = "Your job is to predict the target value based on some features.\n"
            features = X_train.columns
            input_format = "You will be given " + str(len(features)) + " features in total, including: " + ", ".join(features) + ".\n"
            output_format = "Please output the target value as a number. It is very important to only output the target number and nothing else.\n"

            examples = f"You will be given a total of {k} examples\n"
            prompt_note = f"The presented {k} examples above are not in particular order.\n"

            # shuffle column order per example 
            rng = np.random.default_rng(123456) 
            for i in range(k):
                col_order = rng.permutation(features).tolist()              # shuffled feature names
                xrow_shuffled = X_train.iloc[i][col_order]                  # reorder values to match
                examples += (
                    f"Here is example {i+1}:\n"
                    + "A data point has " + narrate_data(col_order, xrow_shuffled, case="json") + "\n"
                    + "The correct target value of this data point is " + str(Y_train.iloc[i]) + ".\n"
                )
            # ----------------------------------------------------

        case _:
            print("Invalid data name.")
            exit()

    return context + input_format + output_format + examples
'''

# query GPT-4o for prediction
# temperature kept at 0 to eliminate randomness
def get_prediction(data_name, X_train, Y_train, new_data):
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": construct_prompt(data_name, X_train, Y_train)},
            {"role": "user", "content": narrate_data(X_train.columns, new_data)}
        ],
        temperature = 0
    )
    return completion.choices[0].message.content


# using batch mode of OpenAI
# prepare batch prediction file
# gpt_model is either a foundation model name or a fine-tuned model ID
# fine_tuned is true or false

def create_batch_prediction(data_name, X_train, Y_train, X_test, gpt_model, fine_tuned, k):
    if fine_tuned:
        f = open("batch_prediction_jobs/" + data_name + "_"+str(k)+ "_fewshot_finetune.jsonl", "w")
    else:
        f = open("batch_prediction_jobs/" + data_name + "_"+str(k)+ "_fewshot_raw.jsonl", "w")
    for i in range(X_test.shape[0]):
        f.write(json.dumps({
            # ID is row number, for tracking
            "custom_id": str(i+1),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": gpt_model,
                "messages": [{
                    "role": "system",
                    "content": construct_prompt(data_name, X_train, Y_train, k)
                }, {
                    "role": "user",
                    "content": narrate_data(X_train.columns, X_test.iloc[i], case="json")
                }],
                "temperature": 0,
                "seed": 123456
            }
        }) + "\n")
    f.close()
    file_name = f"batch_prediction_jobs/" + data_name + "_"+str(k)+ "_fewshot_raw.jsonl"
    logging.info(f"Batch file created: {file_name}")

# code for combination of “Variable Order” and “Format”
'''
def create_batch_prediction(data_name, X_train, Y_train, X_test, gpt_model, fine_tuned, k, shuffle_test_cols=True, seed=123456):
    rng = np.random.default_rng(seed)
    if fine_tuned:
        f = open("batch_prediction_jobs/" + data_name + "_"+str(k)+ "_fewshot_finetune.jsonl", "w")
    else:
        f = open("batch_prediction_jobs/" + data_name + "_"+str(k)+ "_fewshot_raw.jsonl", "w")
    for i in range(X_test.shape[0]):
        if shuffle_test_cols:
            test_col_order = rng.permutation(list(X_train.columns)).tolist()
            x_row = X_test.iloc[i][test_col_order]
            user_content = narrate_data(test_col_order, x_row, case="json")
        else:
            user_content = narrate_data(X_train.columns, X_test.iloc[i], case="json")

        f.write(json.dumps({
            # ID is row number, for tracking
            "custom_id": str(i+1),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": gpt_model,
                "messages": [{
                    "role": "system",
                    "content": construct_prompt(data_name, X_train, Y_train, k)
                }, {
                    "role": "user",
                    "content": user_content
                }],
                "temperature": 0,
                "seed": 123456
            }
        }) + "\n")
    f.close()
    file_name = f"batch_prediction_jobs/" + data_name + "_"+str(k)+ "_fewshot_raw.jsonl"
    logging.info(f"Batch file created: {file_name}")
'''

# launch batch prediction job
# this should only be ran once for each dataset

def launch_batch_prediction(data_name, fine_tuned,k):
    if fine_tuned:
        f = open("batch_prediction_jobs/" + data_name + "_"+str(k)+ "_fewshot_finetune.jsonl", "rb")
    else:
        f = open("batch_prediction_jobs/" + data_name + "_"+str(k)+"_fewshot_raw.jsonl", "rb")
    
    # upload batch file
    batch_input_file = client.files.create(
        file = f,
        purpose = "batch"
    )
    
    # creating the batch
    client.batches.create(
        input_file_id = batch_input_file.id,
        endpoint = "/v1/chat/completions",
        completion_window = "24h",
        metadata = {
            "description": "Batch prediction for " + data_name + " with fine-tuning " + str(fine_tuned)
        } 
    )

#automation
'''
def launch_batch_prediction(data_name, k, fine_tuned=False):
    file_name = f"batch_prediction_jobs/{data_name}_{k}_{'fewshot_finetune' if fine_tuned else 'fewshot_raw'}.jsonl"
    if not os.path.isfile(file_name):
        logging.error("Batch file not found.")
        return

    with open(file_name, "rb") as f:
        try:
            batch_input_file = client.files.create(file=f, purpose="batch")
            batch_job = client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={"description": f"Batch prediction for {data_name} with fine-tuning {fine_tuned}"}
            )
            batch_id = batch_job.id
            logging.info(f"Batch prediction launched successfully with ID: {batch_id}")
            return batch_id
        except Exception as e:
            logging.error(f"Error launching batch prediction: {e}")
            raise
'''

# read results from batch prediction generated file
def read_results(file_obj):
    idx = []
    pred = []
    for line in file_obj:
        idx.append(json.loads(line)["custom_id"])
        pred.append(json.loads(line)["response"]["body"]["choices"][0]["message"]["content"])
    # convert idx to integers and pred to floats, if fail, something is wrong with gpt outputs
    try:
        idx = list(map(int, idx))
        pred = list(map(float, pred))
        return pd.DataFrame({"ID": idx, "Prediction": pred}).sort_values(by = "ID")
    except:
        print("Wrong value types with the batch prediction results.")
        return None
    
def wait_for_completion(job_id, job_type="batch"):
    
    if job_type == "batch":
        poll_interval = 120  # 2 minutes in seconds
        max_wait_time = 3600  # 30 minutes max wait for batch job
    elif job_type == "fine-tune":
        poll_interval = 1200  # 20 minutes in seconds
        max_wait_time = 7200  # 2 hours max wait for fine-tuning
    else:
        logging.error("Invalid job type for status checking.")
        return None

    total_wait_time = 0

    try:
        while total_wait_time < max_wait_time:
            if job_type == "batch":
                result = client.batches.retrieve(job_id)
            elif job_type == "fine-tune":
                result = client.fine_tuning.jobs.retrieve(job_id)
                

            status = result.status
            if status in ["completed", "succeeded"]:
                logging.info(f"{job_type.capitalize()} job {job_id} completed successfully.")
                return result
                break
            elif status in ["failed", "canceled"]:
                logging.error(f"{job_type.capitalize()} job {job_id} failed or was canceled.")
                return None
                break
            else:
                logging.info(f"Waiting for {job_type} job {job_id} to complete. Current status: {status}")

            # Wait for the specified interval
            time.sleep(poll_interval)
            total_wait_time += poll_interval

        if total_wait_time >= max_wait_time:
            logging.warning(f"{job_type.capitalize()} job {job_id} did not complete within the maximum wait time.")
            return None

    except Exception as e:
        logging.error(f"Error checking {job_type} job status: {e}")
        return None

def retrieve_and_save_batch_results(batch_id, data_name, k, fine_tuned=False):
    result_type = "fewshot_finetune" if fine_tuned else "fewshot_raw"
    result = wait_for_completion(batch_id, job_type="batch")
    
    if result is None:
        print(f"Failed to complete batch {batch_id} for {data_name} ({result_type}).")
        return

    # Save batch results
    result_file_content = client.files.content(result.output_file_id)
    file_path = f"prediction_results/" + data_name + "_"+str(k)+ "_fewshot_raw.jsonl"
    with open(file_path, "w") as f:
        f.write(result_file_content.text)
    print(f"Saved {result_type} results for {data_name} to {file_path}.")

# automation
'''
def main():
    #relationship_type = input("Enter relationship type (e.g., 'linear', 'square', 'exp', etc.): ")
    #name = f"synthetic_data_{relationship_type}"
    #data_list = [name]
    data_list = ['synthetic_data_regression_p5']
    k_list = [10,20]
    for data_name in data_list:
        for k in k_list:
            logging.info(f"Processing dataset: {data_name}")

            # Step 1: Launch batch prediction for raw data
            if not os.path.isfile(f"batch_prediction_jobs/" + data_name + "_"+str(k)+ "_fewshot_raw.jsonl"):
                X, Y = read_data(data_name)
                X_train, X_test, Y_train, Y_test = split_data(X, Y)
                create_batch_prediction(data_name, X_train, Y_train, X_test, "gpt-4o-mini", False, k)
                batch_id_raw = launch_batch_prediction(data_name, k, False)
                retrieve_and_save_batch_results(batch_id_raw, data_name, k, fine_tuned=False)

if __name__ == "__main__":
    main()
'''
  


if __name__ == "__main__":
    relationship_type = input("Enter relationship type (e.g., 'linear', 'square', 'exp', etc.): ")
    name = f"synthetic_data_{relationship_type}"
    data_list = [name]
    k = int(input("Enter the number of examples (k) to use for few-shot learning: "))

    # task should generally be performed in the order of 1 to 5
    task = input("Enter Task. \n1: launch batch prediction job \n4: process batch prediction results \n5: report performance \n")
    match task:
        case "1":
            '''
            for data_name in data_list:
                launch_batch_prediction(data_name, False, k)
            '''
            for data_name in data_list:
                if os.path.isfile("batch_prediction_jobs/" + data_name + "_"+str(k)+ "_fewshot_raw.jsonl"):
                    print("Batch prediction job for this dataset already launched.")
                else:
                    X, Y = read_data(data_name)
                    X_train, X_test, Y_train, Y_test = split_data(X, Y)
                    create_batch_prediction(data_name, X_train, Y_train, X_test, "gpt-4o-mini", False, k)
                    launch_batch_prediction(data_name, False, k)
            
        
        case "4":
            batch_id = input("Enter batch ID: \n")
            result = client.batches.retrieve(batch_id)
            if result.status != "completed":
                print("something is wrong with the batch")
                exit()
            else:
                description = result.metadata["description"]
                data_name, fine_tuned = description.split(" ")[3], description.split(" ")[6]
                print(data_name)
                print(fine_tuned)
                #print(data_name, fine_tuned)
                result_file = client.files.content(result.output_file_id)
                # save results to file
                if fine_tuned == "True":
                    f = open("prediction_results/" + data_name + "_"+str(k)+"_fewshot_finetune.jsonl", "w")
                    f.write(result_file.text)
                elif fine_tuned == "False":
                    f = open("prediction_results/" + data_name + "_"+str(k)+"_fewshot_raw.jsonl", "w")
                    f.write(result_file.text)
                else:
                    print("something is wrong with description parsing.")
                    exit()

        case "5":
            for data_name in data_list:
                X, Y = read_data(data_name)
                X_train, X_test, Y_train, Y_test = split_data(X, Y)
                # predictions without fine-tuning
                f_raw = open("prediction_results/" + data_name + "_"+str(k)+ "_fewshot_raw.jsonl", "r")
                pred_raw = read_results(f_raw)
                if pred_raw is not None and pred_raw.shape[0] == X_test.shape[0]:
                    MAE, RMSE, MAPE = performance_eval(pred_raw["Prediction"], Y_test)
                    print(f"LLM performance on fewshot k={k} on {data_name}: MAE: {MAE}, RMSE: {RMSE}, MAPE: {MAPE}")
                else:
                    print("Shape mismatch with the raw prediction results.")
        case _:
            print("Invalid input. Exiting.")
            exit()
