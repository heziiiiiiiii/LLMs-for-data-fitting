import numpy as np
import pandas as pd
import random
from scipy.stats import norm

# ---------- Step 1: build random correlation matrix ----------
def random_correlation_matrix(d, low, high, seed=None):
    rng = np.random.default_rng(seed)

    # Start with identity matrix
    corr = np.eye(d)
    
    # Fill off-diagonal elements with random correlations
    for i in range(d):
        for j in range(i+1, d):
            corr[i, j] = corr[j, i] = rng.uniform(low, high)
    
    # Ensure positive semi-definite by adjusting eigenvalues if needed
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.maximum(eigvals, 1e-8)  # ensure all eigenvalues are positive
    corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    # Re-normalize to ensure diagonal is exactly 1
    D = np.sqrt(np.diag(corr))
    corr = corr / np.outer(D, D)
    np.fill_diagonal(corr, 1.0)

    return corr

# ---------- Step 2: simulate correlated uniforms ----------
def simulate_correlated_uniform(n_samples: int, corr: np.ndarray, seed: int | None = None):
    """
    Simulate n_samples of d correlated Uniform(0,1) variables using a Gaussian copula.
    corr must be a dxd correlation matrix (symmetric, diag=1, positive semidefinite).
    Returns: array shape (n_samples, d)
    """
    rng = np.random.default_rng(seed)

    corr = np.asarray(corr, dtype=float)
    d = corr.shape[0]
    assert corr.shape == (d, d)
    assert np.allclose(np.diag(corr), 1.0)
    assert np.allclose(corr, corr.T)

    # Cholesky method
    L = np.linalg.cholesky(corr)  # add small jitter for numerical stability

    Z = rng.standard_normal(size=(n_samples, d))
    X = Z @ L                   # correlated normals
    U = norm.cdf(X)             # Uniform(0,1) marginals
    return U

def generate_original(n, seed=123456, noise=0, digit=None, column_order=None, column_name=None):
    """
    Original
    f(x) = a·sin(pi/2·x5·x7) + b(5*x0 - e)^2 + c·1/(x2+x9) + d·arctan((x6-x3)/x1) + e*sqrt(x4*x8)
    """
    np.random.seed(seed)
    
    '''
    # Random integer coefficients in [0, 20]
    a, b, c, d, e = np.random.randint(0, 21, size=5)
    print(f"[Params] a={a}, b={b}, c={c}, d={d}, e={e}")
    '''

    np.random.seed(seed) 
    # Generate features
    X = np.random.uniform(0, 1, (n, 10))
    
    # Calculate Y
    Y = (
        8 * np.sin(np.pi/2 * X[:, 5] * X[:, 7]) +
        10 * (X[:, 0] - 0.6) ** 2 +
        np.minimum(4/(X[:, 2] + X[:, 9]), 10) +
        7 * np.arctan((X[:, 6] - X[:, 3])/X[:, 1]) +
        6 * np.sqrt(X[:, 4]*X[:, 8])
    )

    # Add noise
    Y += np.random.normal(0, noise, n)

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


def generate_original_with_correlation(n, seed=123456, noise=0, digit=None, column_order=None, column_name=None):
    """
    Original
    f(x) = a·sin(pi/2·x5·x7) + b(5*x0 - e)^2 + c·1/(x2+x9) + d·arctan((x6-x3)/x1) + e*sqrt(x4*x8)
    """
    d = 10
    corr = random_correlation_matrix(d, low=-0.2, high=0.2, seed=123456)
    X = simulate_correlated_uniform(n_samples=5000, corr=corr, seed=123456)

    X0, X1, X2, X3, X4, X5, X6, X7, X8, X9 = X.T

    Y = (
    8 * np.sin((np.pi / 2) * X5 * X7)
    + 10 * (X0 - 0.6) ** 2
    + np.minimum(4 / (X2 + X9), 10)
    + 7 * np.arctan((X6 - X3) / X1)
    + 6 * np.sqrt(X4 * X8)
    )

    # Add noise
    rng = np.random.default_rng(seed)
    Y += rng.normal(0, noise, size=n)
    #Y += np.random.normal(0, noise, n)
    

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


def generate_all_original(n=5000, seed=123456, noise=0):
    """Generate Friedman datasets with variations using for loops"""
    
    
    # Map dataset names to functions
    func_map = {
        'original_w_corr_w_noise': generate_original_with_correlation
    }
    
    datasets = ['original_w_corr_w_noise']
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
    datasets = generate_all_original(n=5000, seed=123456, noise=noise)