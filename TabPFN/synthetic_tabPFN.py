# simulate 100 synthetic data
import numpy as np
import pandas as pd
import random
from pathlib import Path

def simulate_data(n, p, dgp, distribution, digit, seed=123456, beta=None, noise=None, column_order=None, column_name=None):
    np.random.seed(seed)

    # Generate X
    if distribution == 'normal':
        X = np.random.randn(n, p)
    elif distribution == 'exp':
        X = np.random.exponential(1, (n, p))

    # Generate Y
    if dgp == 'linear':
        Y = X @ beta + noise

    # put X and Y into a dataframe
    data = pd.DataFrame(np.column_stack((X, Y)))

    # Digit control
    if digit == "10_digits":
        data = pd.DataFrame(np.column_stack((X, Y))).round(10)

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

    return data, col_names


def generate_noise(n, noise_seed):
    rng = np.random.default_rng(noise_seed)
    return rng.normal(0, 0.1, n)


def generate_linear_exp_all_positive2_1000x_fixed_beta(
    n=5000,
    p=10,
    n_datasets=100,
    out_dir="TabPFN_data_update/linear_exp_all_positive2",
    base_seed=123456,
    digit=None,
    column_order=None,
    column_name=None,
):

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    beta = [0.92961609, 0.31637555, 0.18391881, 0.20456028, 0.56772503, 0.5955447, 0.96451452, 0.6531771,  0.74890664, 0.65356987]

    rows = []
    all_noise = []
 
    for i in range(n_datasets):
        data_seed = base_seed + i + 1
        noise = generate_noise(n, data_seed)
 
        data, col_names = simulate_data(
            n=n,
            p=p,
            dgp="linear",
            distribution="exp",
            digit=digit,
            seed=data_seed,
            beta=beta,
            noise=noise,
            column_order=column_order,
            column_name=column_name,
        )
    
        data.columns = col_names
 
        file_path = out / f"dataset_{i:04d}.csv"
        data.to_csv(file_path, index=False)
 
        rows.append({"index": i, "data_seed": data_seed, "path": str(file_path)})
        all_noise.append(noise)
 
        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{n_datasets} datasets...")
 
    pd.DataFrame(rows).to_csv(out / "manifest.csv", index=False)
    noise_df = pd.DataFrame(np.array(all_noise).T,columns=[f"dataset_{i:04d}" for i in range(n_datasets)])

    noise_df.to_csv(out / "noise.csv", index=False)
 
    print(f"Wrote {n_datasets} datasets to: {out.resolve()}")


if __name__ == "__main__":
    generate_linear_exp_all_positive2_1000x_fixed_beta(
        n=5000,
        p=10,
        n_datasets=100,
        out_dir="TabPFN_data_update/linear_exp_all_positive2",
        base_seed=123456,
    )
