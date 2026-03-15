import numpy as np
import pandas as pd
import random

def generate_logistic(n, seed=123456, noise=0, digit=None, column_order=None, column_name=None):
    np.random.seed(seed)

    X = np.random.randn(n,10)

    np.random.seed(seed)
    beta = np.random.uniform(0, 1, 10)
    print(beta)
    # [0.12696983 0.96671784 0.26047601 0.89723652 0.37674972 0.33622174 0.45137647 0.84025508 0.12310214 0.5430262 ]
    y_latent = X @ beta

    np.random.seed(seed)
    eps = np.random.normal(0, noise, n)
    y_latent = y_latent + eps

    '''
    # center latent Y to balance classes
    if center_latent:
        y_latent = y_latent - y_latent.mean()
    '''
    
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


def generate_all_logistic(n=5000, seed=123456, noise=0):
    """Generate Friedman datasets with variations using for loops"""
    
    
    # Map dataset names to functions
    func_map = {
        'logistic': generate_logistic
    }
    
    datasets = ['logistic']
    variations = [None, '10_digits', 'shuffle',  "diff_name"]  
    
    # Loop through datasets
    for dataset_name in datasets:
        func = func_map[dataset_name]
        
        # Loop through variations
        for variation in variations:
            
            # Determine which parameter to use
            if variation is None:
                digit, column_order, column_name = None, None, None
            elif variation == '10_digits':
                digit, column_order, column_name = '10_digits', None, None
            elif variation == 'shuffle':
                digit, column_order, column_name = None, 'shuffle', None
            elif variation == 'diff_name':
                digit, column_order, column_name = None, None, "diff_name"
            
            # Generate data
            data, col_names = func(n, seed=seed, noise=noise, digit=digit, 
                                  column_order=column_order, 
                                  column_name=column_name)
            
            # Create filename
            if variation is None:
                filename = f"data/synthetic_data_{dataset_name}.csv"
            else:
                filename = f"data/synthetic_data_{dataset_name}_{variation}.csv"
            
            # Save
            data.to_csv(filename, header=col_names, index=False)
    


if __name__ == "__main__":
    print("Generating Original Benchmark Datasets")
    # Generate all datasets
    noise=0.1
    datasets = generate_all_logistic(n=5000, seed=123456, noise=noise)