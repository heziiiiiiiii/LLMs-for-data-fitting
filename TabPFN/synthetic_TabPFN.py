# simulate synthetic data
import numpy as np
import pandas as pd
import random
from pathlib import Path

# simulate data with n samples and p features
# Y is a linear combination of X with some noise
def simulate_data(n, p, dgp, distribution, digit, seed = 123456, beta=None, column_order=None, column_name=None):
    np.random.seed(seed)

    # Generate X
    if distribution == 'normal':
        X = np.random.randn(n,p)
    elif distribution == 'exp':
        X = np.random.exponential(1, (n, p))

    # Generate Y
    #beta = np.random.uniform(0, 1, p)
    if dgp == 'mean':
        Y = np.mean(X, axis=1)
    elif dgp == 'linear':
        Y = X @ beta + np.random.normal(0, 0.1, n)
        # print(np.random.normal(0, 0.1, n))
    elif dgp == 'sigmoid':
        Y = 1 / (1 + np.exp(-X @ beta)) + np.random.normal(0, 0.1, n)
    #elif dgp == 'sigmoid2':
        #Y = 1 / (1 + np.exp(-X))

    # put X and Y into a dataframe
    data = pd.DataFrame(np.column_stack((X, Y)))
    
    # Digit control
    if digit == "10_digits":
        data = pd.DataFrame(np.column_stack((X, Y))).round(10)
    elif digit == "11_digits":
        data = pd.DataFrame(np.column_stack((X, Y))).round(11)
    elif digit == "12_digits":
        data = pd.DataFrame(np.column_stack((X, Y))).round(12)
    elif digit == "13_digits":
        data = pd.DataFrame(np.column_stack((X, Y))).round(13)
    elif digit == "14_digits":
        data = pd.DataFrame(np.column_stack((X, Y))).round(14)

    # add column names
    col_names = [f"X{i}" for i in range(p)] + ["Y"]

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
        new_name = [f"{ordinal_names[i]}_Variable" if i < len(ordinal_names) else f"{i+1}_th_Variable" for i in range(p)] + ["Y"]
        return data, new_name
    elif column_name == "diff_name2":
        new_name = [f"X0_{i}" for i in range(p)] + ["Y"]
        return data, new_name
    elif column_name == "diff_name3":
        new_name = [f"X0_0_{i}" for i in range(p)] + ["Y"]
        return data, new_name
    elif column_name == "diff_name4":
        new_name = [f"X0_0_0_{i}" for i in range(p)] + ["Y"]
        return data, new_name
    

    return data, col_names

# actually range
def generate_outliers(r, n, seed=12345):
    outlier_values = []
    for j in range(r):
        num_digits = j + 1
        min_value = 10**(num_digits - 1)
        max_value = 10**num_digits - 1
        outlier_values.append(np.random.randint(min_value, max_value + 1, size=n))
    return outlier_values

def format_decimal(x, digits):
    formatted_value = f"{x:.{digits}f}"
    return float(formatted_value)

def generate_combination(n,p, seed=12345):
    dgps = ['linear']
    distributions = ['exp', 'normal']
    #digits = ['fixed_digit', 'random_digit']
    #column_order = 'shuffle'
    #column_name = 'diff_name'

    np.random.seed(seed)
    beta = np.random.uniform(0, 1, p)
    # [0.92961609 0.31637555 0.18391881 0.20456028 0.56772503 0.5955447 0.96451452 0.6531771  0.74890664 0.65356987]
    # outlier_values = generate_outliers(4,n) #range
    
    for dgp in dgps:
        for distribution in distributions:
            data, col_names = simulate_data(n,p,dgp=dgp, distribution=distribution, digit=None, beta=beta, column_order=None, column_name=None）
            file_name = f"data/synthetic_data_{dgp}_{distribution}.csv"
            data.to_csv(file_name, header=col_names, index=False)
            print(f"Generated and saved: {file_name}")

def generate_linear_exp_all_positive2_1000x_perbeta(
    n=5000,
    p=10,
    n_datasets=100,
    out_dir="TabPFN_data/linear_exp_all_positive2",
    base_seed=123456,
    beta_seed_overrides=None, 
):

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if beta_seed_overrides is None:
        beta_seed_overrides = {}

    rows = []
    per_ds_betas = []

    for i in range(n_datasets):
        data_seed = base_seed + i + 1
        # Decide the RNG seed used to draw beta for this dataset
        beta_seed = beta_seed_overrides.get(i, base_seed + 10_000 + i)

        rng_beta = np.random.default_rng(beta_seed)
        beta_i = rng_beta.uniform(0.0, 1.0, size=p)

        data, col_names = simulate_data(
            n=n,
            p=p,
            dgp="linear",
            distribution="exp",
            sign_control="all_positive2",
            scale=None,
            outlier_values=None,
            digit=None,
            seed=data_seed,       
            beta=beta_i,          
            column_order=None,
            column_name=None,
        )

        data.columns = col_names
        file_path = out / f"dataset_{i:04d}.csv"
        data.to_csv(file_path, index=False)

        rows.append({"index": i, "data_seed": data_seed, "beta_seed": beta_seed, "path": str(file_path)})
        per_ds_betas.append({"index": i, "beta_seed": beta_seed, **{f"beta_{j}": beta_i[j] for j in range(p)}})

    # Save logs
    pd.DataFrame(rows).to_csv(out / "manifest.csv", index=False)
    pd.DataFrame(per_ds_betas).to_csv(out / "betas_per_dataset.csv", index=False)

def generate_linear_exp_all_positive2_1000x_perbeta_shuffled(
    n=5000,
    p=10,
    n_datasets=100,
    out_dir="TabPFN_data/linear_exp_all_positive2_shuffle",
    base_seed=123456,
    beta_seed_overrides=None,
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if beta_seed_overrides is None:
        beta_seed_overrides = {}

    rows = []
    per_ds_betas = []
    column_orders = []

    for i in range(n_datasets):
        data_seed = base_seed + i + 1
        # Decide the RNG seed used to draw beta for this dataset
        beta_seed = beta_seed_overrides.get(i, base_seed + 10_000 + i)

        rng_beta = np.random.default_rng(beta_seed)
        beta_i = rng_beta.uniform(0.0, 1.0, size=p)

        # Generate data with shuffled column order
        data, col_names = simulate_data(
            n=n,
            p=p,
            dgp="linear",
            distribution="exp",
            sign_control="all_positive2",
            scale=None,
            outlier_values=None,
            digit=None,
            seed=data_seed,      
            beta=beta_i,          
            column_order="shuffle",  
            column_name=None,
        )

        data.columns = col_names
        
        # Save the file
        file_path = out / f"dataset_{i:04d}.csv"
        data.to_csv(file_path, index=False)

        # Track the column shuffle order
        original_order = [f"X{j}" for j in range(p)] + ["Y"]
        shuffled_order = col_names.copy()
        
        rows.append({
            "index": i, 
            "data_seed": data_seed, 
            "beta_seed": beta_seed, 
            "path": str(file_path),
            "column_shuffle_order": str(shuffled_order)
        })
        
        per_ds_betas.append({
            "index": i, 
            "beta_seed": beta_seed, 
            **{f"beta_{j}": beta_i[j] for j in range(p)}
        })
        
        column_orders.append({
            "index": i,
            "original_order": str(original_order),
            "shuffled_order": str(shuffled_order)
        })

    # Save logs
    pd.DataFrame(rows).to_csv(out / "manifest.csv", index=False)
    pd.DataFrame(per_ds_betas).to_csv(out / "betas_per_dataset.csv", index=False)
    pd.DataFrame(column_orders).to_csv(out / "column_orders.csv", index=False)

def generate_linear_exp_all_positive2_1000x_perbeta_10digits(
    n=5000,
    p=10,
    n_datasets=100,
    out_dir="TabPFN_data/linear_exp_all_positive2_10_digits",
    base_seed=123456,
    beta_seed_overrides=None,
):

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if beta_seed_overrides is None:
        beta_seed_overrides = {}

    rows = []
    per_ds_betas = []

    for i in range(n_datasets):
        data_seed = base_seed + i + 1
        # Decide the RNG seed used to draw beta for this dataset
        beta_seed = beta_seed_overrides.get(i, base_seed + 10_000 + i)

        rng_beta = np.random.default_rng(beta_seed)
        beta_i = rng_beta.uniform(0.0, 1.0, size=p)

        # Generate data with 10-digit precision
        data, col_names = simulate_data(
            n=n,
            p=p,
            dgp="linear",
            distribution="exp",
            sign_control="all_positive2",
            scale=None,
            outlier_values=None,
            digit="10_digits",   
            seed=data_seed,       
            beta=beta_i,          
            column_order=None,
            column_name=None,
        )

        # Set column names
        data.columns = col_names
        
        # Save the file
        file_path = out / f"dataset_{i:04d}.csv"
        data.to_csv(file_path, index=False)

        rows.append({
            "index": i, 
            "data_seed": data_seed, 
            "beta_seed": beta_seed, 
            "path": str(file_path),
            "precision": "10_digits"
        })
        
        per_ds_betas.append({
            "index": i, 
            "beta_seed": beta_seed, 
            **{f"beta_{j}": beta_i[j] for j in range(p)}
        })

    # Save logs
    pd.DataFrame(rows).to_csv(out / "manifest.csv", index=False)
    pd.DataFrame(per_ds_betas).to_csv(out / "betas_per_dataset.csv", index=False)



if __name__ == "__main__":
    '''
    generate_linear_exp_all_positive2_1000x_perbeta(
        n=5000,
        p=10,
        n_datasets=100,
        out_dir="TabPFN_data/linear_exp_all_positive2",
        base_seed=123456,
        beta_seed_overrides=None,
    )
    
    generate_linear_exp_all_positive2_1000x_perbeta_shuffled(
    n=5000,
    p=10,
    n_datasets=100,
    out_dir="TabPFN_data/linear_exp_all_positive2_shuffle",
    base_seed=123456,
    beta_seed_overrides=None,
    )
    '''

    generate_linear_exp_all_positive2_1000x_perbeta_10digits(
    n=5000,
    p=10,
    n_datasets=100,
    out_dir="TabPFN_data/linear_exp_all_positive2_10_digits",
    base_seed=123456,
    beta_seed_overrides=None,
)

