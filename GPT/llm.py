from openai import OpenAI
import pandas as pd
import numpy as np
from utility import *
import time
import json
import os.path
import math

client = OpenAI(organization = "xxx", project = "xxx")

seed = 123456

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
def construct_prompt(data_name, X_train, Y_train):
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
            example = "Here is an example:\n" + "A data point has " + narrate_data(features, X_train.iloc[0], case="json") + "\nThe correct target value of this data point is " + str(Y_train.iloc[0])

        case "laptop_noduplicates":
            context = "Your job is to predict the target value based on some features.\n"
            features = X_train.columns
            input_format = "You will be given " + str(len(features)) + " features in total, including: " + ", ".join(features) + ".\n"
            output_format = "Please output the target value as a number. It is very important to only output the target number and nothing else.\n"
            example = "Here is an example:\n" + "A data point has " + narrate_data(features, X_train.iloc[0]) + "\nThe correct target value of this data point is " + str(Y_train.iloc[0])

        case "multiplication":
            context = "You are a calculator.\n"
            features = X_train.columns  
            input_format = "You will be given two numbers: " + ", ".join(features) + ".\n"
            output_format = "Please output only the multiplication result as a number. Respond only with the numeric result, without any explanation or steps.\n"

        case _:
            print("Invalid data name.")
            exit()

    #return context + input_format + output_format + example
    return context + input_format + output_format


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

def create_batch_prediction(data_name, X_train, Y_train, X_test, gpt_model, fine_tuned):
    if fine_tuned:
        f = open("batch_prediction_jobs/" + data_name + "_finetune.jsonl", "w")
    else:
        f = open("batch_prediction_jobs/" + data_name + "_raw.jsonl", "w")
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
                    "content": construct_prompt(data_name, X_train, Y_train)
                }, {
                    "role": "user",
                    "content": narrate_data(X_train.columns, X_test.iloc[i], case='base')
                }],
                "temperature": 0,
                "seed": seed 
            }
        }) + "\n")
    f.close()

# code for combination of “Variable Order” and “Format”
'''
def create_batch_prediction(data_name, X_train, Y_train, X_test, gpt_model, fine_tuned, shuffle_test_cols=True, seed=123456):
    rng = np.random.default_rng(seed)
    if fine_tuned:
        f = open("batch_prediction_jobs/" + data_name + "_finetune.jsonl", "w")
    else:
        f = open("batch_prediction_jobs/" + data_name + "_raw.jsonl", "w")
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
                    "content": construct_prompt(data_name, X_train, Y_train)
                }, {
                    "role": "user",
                    "content": user_content
                }],
                "temperature": 0,
                "seed": 123456
            }
        }) + "\n")
    f.close()
'''

# launch batch prediction job
# this should only be ran once for each dataset!!!
def launch_batch_prediction(data_name, fine_tuned):
    if fine_tuned:
        f = open("batch_prediction_jobs/" + data_name + "_finetune.jsonl", "rb")
    else:
        f = open("batch_prediction_jobs/" + data_name + "_raw.jsonl", "rb")
    
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


# in case of fine-tuning, also need to create the fine-tuning training data and launch the job

def create_finetune_training(data_name, X_train, Y_train):
    f = open("finetune_training_data/" + data_name + ".jsonl", "w")
    for i in range(X_train.shape[0]):
        f.write(json.dumps({
            "messages": [{
                "role": "system",
                "content": construct_prompt(data_name, X_train, Y_train)
            }, {
                "role": "user",
                "content": narrate_data(X_train.columns, X_train.iloc[i], case='json')
            }, {
                "role": "assistant",
                "content": str(Y_train.iloc[i])
            }]
        }) + "\n")
    f.close()

# code for combination of “Variable Order” and “Format”
'''
def create_finetune_training(data_name, X_train, Y_train, seed=123456):
    rng = np.random.default_rng(seed)
    f = open("finetune_training_data/" + data_name + ".jsonl", "w")
    for i in range(X_train.shape[0]):
        shuffled_cols = rng.permutation(list(X_train.columns)).tolist()
        x_row = X_train.iloc[i][shuffled_cols]
        f.write(json.dumps({
            "messages": [{
                "role": "system",
                "content": construct_prompt(data_name, X_train, Y_train)
            }, {
                "role": "user",
                "content": narrate_data(shuffled_cols, x_row, case='json')
            }, {
                "role": "assistant",
                "content": str(Y_train.iloc[i])
            }]
        }) + "\n")
    f.close()
'''

# launch fine-tune job
# this should only be ran once for each dataset
def launch_finetune_job(data_name):
    f = open("finetune_training_data/" + data_name + ".jsonl", "rb")
    
    # upload training data
    training_data_file = client.files.create(
        file = f,
        purpose = "fine-tune"
    )

    # create fine-tuning job
    client.fine_tuning.jobs.create(
        training_file = training_data_file.id,
        # so far can only fine-tune 4o-mini. Fine-tune 4o is by request only
        model="gpt-4o-mini-2024-07-18",
        seed=seed
    )

    '''
    # create fine-tuning job
    client.fine_tuning.jobs.create(
        training_file = training_data_file.id,
        # Fine-tune 4o is by request only
        model="gpt-4o-2024-08-06"
    )
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
    

if __name__ == "__main__":
    #relationship_type = input("Enter relationship type (e.g., 'linear', 'square', 'exp', etc.): ")
    #name = f"synthetic_data_{relationship_type}"
    name = input("Enter name: ")
    data_list = [name]

    # task should generally be performed in the order of 1 to 5
    task = input("Enter Task. \n1: launch batch prediction job \n2: launch fine-tuning job \n3: launch batch prediction job with fine-tuned model \n4: process batch prediction results \n5: report performance \n")
    match task:
        case "1":
            for data_name in data_list:
                if os.path.isfile("batch_prediction_jobs/" + data_name + "_raw.jsonl"):
                    print("Batch prediction job for this dataset already launched.")
                else:
                    X, Y = read_data(data_name)
                    X_train, X_test, Y_train, Y_test = split_data(X, Y)
                    create_batch_prediction(data_name, X_train, Y_train, X_test, "gpt-4o", False)
                    #create_batch_prediction(data_name, X_train, Y_train, X_test, "gpt-4o-mini", False)
                    launch_batch_prediction(data_name, False)
        
        case "2":
            for data_name in data_list:
                if os.path.isfile("finetune_training_data/" + data_name + ".jsonl"):
                    print("Fine-tune already done for this dataset.")
                else:
                    X, Y = read_data(data_name)
                    X_train, X_test, Y_train, Y_test = split_data(X, Y)
                    #X_train, X_test, Y_train, Y_test = split_missing_data(X, Y)
                    create_finetune_training(data_name, X_train, Y_train)
                    launch_finetune_job(data_name)
        
        case "3":
            finetune_id = input("Enter fine-tuned job ID: \n")
            result = client.fine_tuning.jobs.retrieve(finetune_id)
            if result.error.code != None:
                print("something is wrong with the fine-tuning job")
                exit()
            else:
                finetune_model = result.fine_tuned_model
                # now launch batch prediction using fine-tuned model
                for data_name in data_list:
                    if os.path.isfile("batch_prediction_jobs/" + data_name + "_finetune.jsonl"):
                        print("Batch prediction job for this dataset already launched.")
                    else:
                        X, Y = read_data(data_name)
                        X_train, X_test, Y_train, Y_test = split_data(X, Y)
                        #X_train, X_test, Y_train, Y_test = split_missing_data(X, Y)
                        create_batch_prediction(data_name, X_train, Y_train, X_test, finetune_model, True)
                        launch_batch_prediction(data_name, True)
        
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
                    f = open("prediction_results/" + data_name + "_finetune.jsonl", "w")
                    f.write(result_file.text)
                elif fine_tuned == "False":
                    f = open("prediction_results/" + data_name + "_raw.jsonl", "w")
                    f.write(result_file.text)
                else:
                    print("something is wrong with description parsing.")
                    exit()

        case "5":
            for data_name in data_list:
                X, Y = read_data(data_name)
                X_train, X_test, Y_train, Y_test = split_data(X, Y)
                #X_train, X_test, Y_train, Y_test = split_missing_data(X, Y)
                # predictions without fine-tuning
                
                '''
                f_raw = open("prediction_results/" + data_name + "_raw.jsonl", "r")
                pred_raw = read_results(f_raw)
                if pred_raw is not None and pred_raw.shape[0] == X_test.shape[0]:
                    MAE, RMSE, MAPE = performance_eval(pred_raw["Prediction"], Y_test)
                    print(f"LLM performance without fine-tuning on {data_name}: MAE: {MAE}, RMSE: {RMSE}, MAPE: {MAPE}")
                else:
                    print("Shape mismatch with the raw prediction results.")
                '''
                
                # predictions with fine-tuning
                f_finetune = open("prediction_results/" + data_name + "_finetune.jsonl", "r")
                pred_finetune = read_results(f_finetune)
                if pred_finetune is not None and pred_finetune.shape[0] == X_test.shape[0]:
                    #output = performance_eval(pred_finetune["Prediction"], Y_test, X_test)
                    #print(output)
                    MAE, RMSE, MAPE = performance_eval(pred_finetune["Prediction"], Y_test)
                    print(f"LLM performance with fine-tuning on {data_name}: MAE: {MAE}, RMSE: {RMSE}, MAPE: {MAPE}")
                else:
                    print("Shape mismatch with the fine-tuned prediction results.")
        
        case _:
            print("Invalid input. Exiting.")

            exit()
