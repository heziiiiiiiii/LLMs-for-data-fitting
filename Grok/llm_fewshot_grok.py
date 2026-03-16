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
    timeout=3600, 
)

class NumericPrediction(BaseModel):
    prediction: float 

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
    match data_name:
        case synthetic_name if synthetic_name.startswith("synthetic_"):
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

        case _:
            print("Invalid data name.")
            exit()

    return context + input_format + output_format + examples


def sequential_prediction_xai(data_name, X_train, Y_train, X_test, Y_test, model_name, k):
    
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
            chat.append(user(narrate_data(X_train.columns, X_test.iloc[i], case="base")))
            
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
    relationship_types = [] #input data name
    k_list = [] #input number of fewshots
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

