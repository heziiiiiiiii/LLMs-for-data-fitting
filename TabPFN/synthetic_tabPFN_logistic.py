import numpy as np
import pandas as pd
import random
from pathlib import Path
from scipy.stats import norm

def generate_logistic(n, seed=123456, noise=0, digit=None, column_order=None, column_name=None):
    np.random.seed(seed)

    X = np.random.randn(n,10)

    beta = [0.12696983, 0.96671784, 0.26047601, 0.89723652, 0.37674972, 0.33622174, 0.45137647, 0.84025508, 0.12310214, 0.5430262 ]
    y_latent = X @ beta

    np.random.seed(seed)
    eps = np.random.normal(0, noise, n)
    y_latent = y_latent + eps

    # probability
    p = 1 / (1 + np.exp(-y_latent))

    # Bernoulli sampling
    Y = np.random.binomial(1, p)

    data = pd.DataFrame(np.column_stack((X, Y)))

    # Digit control
    if digit == "10_digits":
        data = pd.DataFrame(np.column_stack((X,Y))).round(10)

    # add column names
    col_names = [f"X{i}" for i in range(10)] + ["Y"]

    # Column order
    if column_order == "shuffle":
        X_columns = data.iloc[:, :-1]
        random.seed(seed)
        shuffled_indices = random.sample(range(len(X_columns.columns)), len(X_columns.columns))
        shuffled_columns = [X_columns.columns[i] for i in shuffled_indices]
        shuffled_data = data[shuffled_columns + [data.columns[-1]]]
        new_col_name = [f"X{i}" for i in shuffled_indices] + ["Y"]
        return shuffled_data, new_col_name
    
    # Column name
    if column_name == "diff_name":
        ordinal_names = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eighth", "Ninth", "Tenth",
                     "Eleventh", "Twelfth", "Thirteenth", "Fourteenth", "Fifteenth", "Sixteenth", "Seventeenth", "Eighteenth", "Nineteenth", "Twentieth"]
        new_name = [f"{ordinal_names[i]}_Variable" if i < len(ordinal_names) else f"{i+1}_th_Variable" for i in range(10)] + ["Y"]
        return data, new_name

    return data, col_names

def generate_all_logistic_100x(n=5000, n_datasets=100, base_seed=123456):
    # Map to functions
    func_map = {
        'logistic': generate_logistic
    }
    
    # Define all combinations
    data_types = ['logistic']
    variations = {
        'base': {'digit': None, 'column_order': None, 'column_name': None},
        '10_digits': {'digit': '10_digits', 'column_order': None, 'column_name': None},
        'shuffle': {'digit': None, 'column_order': 'shuffle', 'column_name': None}
    }
    
    # Loop through all combinations
    for type in data_types:
        func = func_map[type]
        
        for var_name, var_params in variations.items():
            # Create output directory
            if var_name == 'base':
                out_dir = f"TabPFN_data_update/{type}"
            else:
                out_dir = f"TabPFN_data_update/{type}_{var_name}"
            
            out = Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            
            print(f"Generating {n_datasets} datasets for {type}_{var_name}...")
            
            rows = []
            column_orders_log = []
            
            # Generate 100 datasets
            for i in range(n_datasets):
                data_seed = base_seed + i + 1
                
                # Generate data
                data, col_names = func(
                    n=n,
                    seed=data_seed,
                    noise=0.1,
                    digit=var_params['digit'],
                    column_order=var_params['column_order']
                )
                
                data.columns = col_names
                
                # Save file
                file_path = out / f"dataset_{i:04d}.csv"
                data.to_csv(file_path, index=False)
                
                # Log metadata
                rows.append({
                    "index": i,
                    "data_seed": data_seed,
                    "path": str(file_path),
                    "friedman_type": type,
                    "variation": var_name
                })
                
                # Track column order if shuffled
                if var_params['column_order'] == 'shuffle':
                    p = len(col_names) - 1
                    original_order = [f"X{j}" for j in range(p)] + ["Y"]
                    column_orders_log.append({
                        "index": i,
                        "original_order": str(original_order),
                        "shuffled_order": str(col_names)
                    })
            
            # Save logs
            pd.DataFrame(rows).to_csv(out / "manifest.csv", index=False)
            if column_orders_log:
                pd.DataFrame(column_orders_log).to_csv(out / "column_orders.csv", index=False)
            
            print(f"Saved to {out_dir}")


if __name__ == "__main__":
    generate_all_logistic_100x(n=5000, n_datasets=100, base_seed=123456)
