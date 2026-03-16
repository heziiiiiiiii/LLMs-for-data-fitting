from anthropic import Anthropic
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

client = Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),  
)

PRED_SCHEMA = {
    "type": "object",
    "properties": {
        "prediction": {"type": "number"}
    },
    "required": ["prediction"],
    "additionalProperties": False
}

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

def construct_prompt(data_name, X_train, Y_train, k):
    if data_name.startswith("synthetic_"):
        context = "Your job is to predict the target value based on some features.\n"
        features = X_train.columns
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

    else:
        print("Invalid data name.")
        exit()

    return context + input_format + output_format + examples


def launch_batch_prediction_claude_structured(data_name, X_train, Y_train, X_test, model_name, k):
    sys_prompt = construct_prompt(data_name, X_train, Y_train, k)
    features = list(X_train.columns)

    requests = []
    for i in range(X_test.shape[0]):
        custom_id = str(i + 1)
        user_content = narrate_data(features, X_test.iloc[i], case="base")

        requests.append({
            "custom_id": custom_id,
            "params": {
                "model": model_name,
                "temperature": 0,
                "max_tokens": 64,
                "system": sys_prompt,
                "messages": [{"role": "user", "content": user_content}],
                "output_config": {
                    "format": {
                        "type": "json_schema",
                        "schema": PRED_SCHEMA
                    }
                }
            }
        })

    mb = client.messages.batches.create(requests=requests)
    logging.info(f"Created batch: {mb.id}")
    return mb.id

def wait_for_batch_completion(batch_id, poll_interval=30, max_wait_time=24*3600):
    waited = 0
    while waited < max_wait_time:
        b = client.messages.batches.retrieve(batch_id)
        rc = b.request_counts
        logging.info(
            f"Batch {b.id}: status={b.processing_status} | "
            f"processing={rc.processing}, succeeded={rc.succeeded}, errored={rc.errored}, "
            f"canceled={rc.canceled}, expired={rc.expired}"
        )
        if b.processing_status == "ended":
            return b
        time.sleep(poll_interval)
        waited += poll_interval
    return None


def retrieve_batch_results_claude(batch_id, Y_test):
    rows = []
    failed = []

    iterator = client.messages.batches.results(batch_id)

    for item in iterator:
        rtype = item.result.type

        if rtype == "succeeded":
            msg = item.result.message
            text = msg.content[0].text if msg.content else ""

            try:
                obj = json.loads(text)
                pred = float(obj["prediction"])

                idx = int(item.custom_id) - 1   # custom_id = 1-based index
                true_val = float(Y_test.iloc[idx])

                rows.append((int(item.custom_id), pred, true_val))

            except Exception as e:
                failed.append((item.custom_id, f"parse_error: {e}", text, msg.stop_reason))

        elif rtype == "errored":
            failed.append((item.custom_id, "errored", str(item.result.error), None))
        else:
            failed.append((item.custom_id, rtype, None, None))

    df = (
        pd.DataFrame(rows, columns=["ID", "Prediction", "True_value"])
        .sort_values("ID")
        .reset_index(drop=True)
    )

    return df, failed

            
if __name__ == "__main__":
    relationship_type = input("Enter relationship type (e.g., 'linear' etc.): ")
    data_name = f"synthetic_data_{relationship_type}"
    k = int(input("Enter the number of examples (k) to use for few-shot learning: "))
    model_name = "claude-sonnet-4-5"

    # Make sure output folder exists
    os.makedirs("prediction_results", exist_ok=True)

    out_csv = f"prediction_results/{data_name}_{k}_fewshot_claude.csv"

    if os.path.isfile(out_csv):
        print(f"Predictions already exist: {out_csv}")
        exit()

    # Load data
    X, Y = read_data(data_name)
    X_train, X_test, Y_train, Y_test = split_data(X, Y)

    batch_id = launch_batch_prediction_claude_structured(
        data_name, X_train, Y_train, X_test, model_name, k
    )

    b = wait_for_batch_completion(batch_id, poll_interval=30)
    if b is None:
        raise RuntimeError("Batch did not finish in time")

    df, failed = retrieve_batch_results_claude(batch_id, Y_test)

    os.makedirs("prediction_results", exist_ok=True)
    out_csv = f"prediction_results/{data_name}_{k}_fewshot_claude.csv"
    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)

    if failed:
        print(f"Failed/odd outputs: {len(failed)} (showing first 5)")
        for x in failed[:5]:
            print(x)

