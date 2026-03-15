from xai_sdk import Client
from xai_sdk.chat import system, user
from pydantic import BaseModel, Field
import os
import pandas as pd
import numpy as np
from utility import *
import logging
import time
import json
import os.path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

client = Client(
    api_key=os.getenv("XAI_API_KEY"),
    timeout=3600, # Override default timeout with longer timeout for reasoning models
)

class NumericPrediction(BaseModel):
    prediction: float 


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
# system prompt consists of task, input format, output format, and one example constructed from the first training example

def construct_prompt(data_name, X_train, Y_train, k):
    match data_name:
        case synthetic_name if synthetic_name.startswith("synthetic_"):
            context = "Your job is to predict the target value based on some features.\n"
            features = X_train.columns
            #new_column_names = [f"X{i}" for i in range(10)]
            input_format = "You will be given " + str(len(features)) + " features in total, including: " + ", ".join(features) + ".\n"
            output_format = "Please output the target value as a number. It is very important to only output the target number and nothing else.\n"
            examples = f"You will be given a total of {k} examples\n"
            #for i in reversed(range(k)):
            for i in range(k):
                examples += (
                    #f"Here is example {k - i}:\n" +
                    f"Here is example {i+1}:\n" +
                    "A data point has " + narrate_data(features, X_train.iloc[i], case="json") + "\n" +
                    "The correct target value of this data point is " + str(Y_train.iloc[i]) + ".\n"
                )

        case _:
            print("Invalid data name.")
            exit()

    return context + input_format + output_format + examples

# using batch mode of OpenAI
# prepare batch prediction file
# gpt_model is either a foundation model name or a fine-tuned model ID
# fine_tuned is true or false

'''
def launch_batch_prediction_xai(data_name, X_train, Y_train, X_test, model_name, k, batch_name):
    batch_name = f"{data_name}_k{k}_fewshot"

    batch = client.batch.create(batch_name=batch_name) 
    batch_id = batch.batch_id
    logging.info(f"Created xAI batch: {batch_name} ({batch_id})")

    sys_prompt = construct_prompt(data_name, X_train, Y_train, k)

    batch_requests = []
    for i in range(X_test.shape[0]):
        req_id = str(i + 1)  

        chat = client.chat.create(
            model="grok-4-1-fast-non-reasoning",
            batch_request_id=req_id,
            temperature=0,
        )
        chat.append(system(sys_prompt))
        chat.append(user(narrate_data(X_train.columns, X_test.iloc[i], case="json")))
        batch_requests.append(chat)

    client.batch.add(batch_id=batch_id, batch_requests=batch_requests)  
    logging.info(f"Added {len(batch_requests)} requests to batch {batch_id}")

    return batch_id


def wait_for_batch_completion_xai(batch_id, poll_interval=5, max_wait_time=3600):

    total_wait = 0

    while total_wait < max_wait_time:
        batch = client.batch.get(batch_id=batch_id)  
        pending = batch.state.num_pending
        done = batch.state.num_success + batch.state.num_error

        logging.info(f"Batch {batch_id}: {done}/{batch.state.num_requests} processed; pending={pending}")

        if pending == 0:
            return batch

        time.sleep(poll_interval)
        total_wait += poll_interval

    logging.warning(f"Batch {batch_id} did not finish within max_wait_time={max_wait_time}s")
    return None

def retrieve_batch_results_xai(batch_id, limit=200):

    all_succeeded = []
    all_failed = []
    pagination_token = None

    while True:
        page = client.batch.list_batch_results(
            batch_id=batch_id,
            limit=limit,
            pagination_token=pagination_token,
        )  # :contentReference[oaicite:8]{index=8}

        all_succeeded.extend(page.succeeded)
        all_failed.extend(page.failed)

        if page.pagination_token is None:
            break
        pagination_token = page.pagination_token

    return all_succeeded, all_failed


def results_to_dataframe_xai(all_succeeded):
    ids = []
    preds = []

    for r in all_succeeded:
        ids.append(int(r.batch_request_id))
        preds.append(r.response.content)

    # convert predictions to float
    try:
        preds = list(map(float, preds))
    except Exception:
        raise ValueError("Some outputs were not pure numbers. Check prompt/output constraints.")

    return pd.DataFrame({"ID": ids, "Prediction": preds}).sort_values("ID")
'''

def sequential_prediction_xai(data_name, X_train, Y_train, X_test, Y_test, model_name, k):
    """Send requests sequentially instead of using batch mode"""
    
    sys_prompt = construct_prompt(data_name, X_train, Y_train, k)
    
    predictions = []
    
    for i in range(X_test.shape[0]):
        try:
            # Create a new chat for each test point
            chat = client.chat.create(
                model=model_name,
                temperature=0,
            )
            chat.append(system(sys_prompt))
            chat.append(user(narrate_data(X_train.columns, X_test.iloc[i], case="json")))
            
            '''
            response, parsed = chat.parse(NumericPrediction)
            pred = float(parsed.prediction)
            '''
            response = chat.sample()
            pred = response.content
            true_val = float(Y_test.iloc[i])
            
            predictions.append({
                "ID": i + 1,
                "Prediction": pred,
                "True_value": true_val,
            })
            
            if (i + 1) % 10 == 0:
                logging.info(f"Completed {i + 1}/{X_test.shape[0]} predictions")
            
        except Exception as e:
            logging.error(f"Error on prediction {i + 1}: {str(e)}")
            predictions.append({
                "ID": i + 1,
                "Prediction": None,
                "True_value": float(Y_test.iloc[i]) if i < len(Y_test) else None,
            })
    
    return pd.DataFrame(predictions)
            
if __name__ == "__main__":
    '''
    relationship_type = input("Enter relationship type (e.g., 'linear', 'square', 'exp', etc.): ")
    data_name = f"synthetic_data_{relationship_type}"
    k = int(input("Enter the number of examples (k) to use for few-shot learning: "))
    '''
    relationship_types = ['logistic_dicFormat']
    k_list = [500]
    # Make sure output folder exists
    os.makedirs("prediction_results", exist_ok=True)

    for relationship_type in relationship_types:
        data_name = f"synthetic_data_{relationship_type}"

        # Load data
        X, Y = read_data(data_name)
        X_train, X_test, Y_train, Y_test = split_data(X, Y)

        for k in k_list:
            out_csv = f"prediction_results/{data_name}_{k}_fewshot_grok.csv"

            if os.path.isfile(out_csv):
                print(f"Predictions already exist: {out_csv}")
                continue

            print(f"RUN {data_name}, k = {k}")

            df_pred = sequential_prediction_xai(
            data_name=data_name,
            X_train=X_train,
            Y_train=Y_train,
            X_test=X_test,
            Y_test=Y_test,
            model_name="grok-4-1-fast-non-reasoning",
            k=k
        )
            # Save CSV predictions
            df_pred.to_csv(out_csv, index=False)
            print(f"Saved predictions to: {out_csv}")



    '''
    # Launch xAI Grok batch
    batch_id = launch_batch_prediction_xai(
        data_name=data_name,
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        model_name="grok-4-1-fast-non-reasoning",
        k=k,
        batch_name=f"Batch prediction for {data_name} k={k}",
    )

    # Wait for completion
    batch = wait_for_batch_completion_xai(batch_id, poll_interval=5, max_wait_time=3600)
    if batch is None:
        raise RuntimeError(f"Batch {batch_id} did not complete in time.")

    # Retrieve results (with pagination)
    succeeded, failed = retrieve_batch_results_xai(batch_id)
    if failed:
        logging.warning(f"{len(failed)} requests failed in batch {batch_id}.")
        # Optional: print a few failure IDs
        logging.warning("First few failures: " + ", ".join([f.batch_request_id for f in failed[:10]]))

    # Convert to DF
    df_pred = results_to_dataframe_xai(succeeded)
    '''

