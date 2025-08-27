# simulate synthetic data
import numpy as np
import pandas as pd
import random

# simulate data with n samples and p features
# Y is a linear combination of X with some noise
def simulate_data(n, p, dgp, distribution, sign_control, scale, outlier_values, digit, seed = 123456, beta=None, column_order=None, column_name=None):
    np.random.seed(seed)

    # Generate X
    if distribution == 'uniform':
        X = np.random.rand(n,p)
    elif distribution == 'normal':
        X = np.random.randn(n,p)
    elif distribution == 'exp':
        X = np.random.exponential(1, (n, p))

    # Outliers control (Scale)
    if scale == 'random_gradual_outliers':
        for i in range(n):
            col_indices = np.random.choice(X.shape[1], size=4, replace=False)
            for j, col in enumerate(col_indices):  # 4 outliers
                X[i, col] = outlier_values[j][i] + (X[i, col] % 1)
    elif scale == 'random_outliers':
        for i in range(n):
            num_outliers = np.random.randint(1, p//2)  # at least 1 outlier and at most p
            col_indices = np.random.choice(X.shape[1], size=num_outliers, replace=False)
            for col in col_indices:
                category = np.random.choice([1, 2, 3])
                if category == 1:
                    outlier_magnitude = np.random.randint(1, 10)      
                elif category == 2:
                    outlier_magnitude = np.random.randint(10, 100)    
                elif category == 3:
                    outlier_magnitude = np.random.randint(100, 1000)
                #outlier_magnitude = np.random.randint(1, 10000)  
                X[i, col] = outlier_magnitude + (X[i, col] % 1)
    elif scale == 'fixed_outlier':
        for i in range(n):
            outlier_value = np.random.randint(100,1000)
            X[i,0] = outlier_value + (X[i,1] % 1)
    elif scale == 'fixed_scale':
        for i in range(n):
            X[i,3] *= 100

    
    # Sign control
    if sign_control == 'all_positive2':
        X = np.abs(X)
    elif sign_control == 'all_positive_large_scale':
        X = np.abs(X)
        X = 100*X
    elif sign_control == 'all_negative2':
        X = -np.abs(X)
    elif sign_control == 'one_negative_fixed2':
        X[:, 0] = -np.abs(X[:,0])
        X[:, 1:] = np.abs(X[:, 1:])
    elif sign_control == 'one_negative_random2':
        for i in range(n):
            neg_pos = np.random.randint(0,p)
            X[i,:] = np.abs(X[i,:])
            X[i, neg_pos] = -(X[i, neg_pos])
    elif sign_control == 'random_negative_fixed2':
        sign = np.random.choice([1, -1], size=p, p=[0.5, 0.5])
        X = np.abs(X) * sign
    elif sign_control == 'random_negative_random2':
        sign = np.random.choice([1, -1], size=(n, p), p=[0.5, 0.5])
        X = np.abs(X) * sign

    # Digit control
    if digit == 'fixed_digit':
        digits = list(range(1, p + 1))
        for i in range(n):
            X[i] = [format_decimal(X[i][j], digits[j]) for j in range(p)]
    elif digit == 'random_digit':
        base_digits = list(range(1, p + 1))
        for i in range(n):
            digits = base_digits.copy()
            random.shuffle(digits)
            X[i] = [format_decimal(X[i][j], digits[j]) for j in range(p)]
    elif digit == "10_digit_forX":
        X = np.round(X, 10)


    # Generate Y
    #beta = np.random.uniform(0, 1, p)
    if dgp == 'mean':
        Y = np.mean(X, axis=1)
    elif dgp == 'linear':
        Y = X @ beta + np.random.normal(0, 0.1, n)
        #Y = X @ beta + 2*np.random.normal(0, 0.1, n)
        print(np.random.normal(0, 0.1, n))
    elif dgp == 'sigmoid':
        Y = 1 / (1 + np.exp(-X @ beta)) + np.random.normal(0, 0.1, n)
    #elif dgp == 'sigmoid2':
        #Y = 1 / (1 + np.exp(-X))


    # put X and Y into a dataframe
    data = pd.DataFrame(np.column_stack((X, Y)))
    # add column names
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
    col_names = [f"X{i}" for i in range(p)] + ["Y"]
    print(len(col_names), data.shape[1])

    
    if column_order == "shuffle":
        X_columns = data.iloc[:, :-1]
        random.seed(seed)
        shuffled_indices = random.sample(range(len(X_columns.columns)), len(X_columns.columns))
        shuffled_columns = [X_columns.columns[i] for i in shuffled_indices]
        shuffled_data = data[shuffled_columns + [data.columns[-1]]]
        new_col_name = [f"X{i}" for i in shuffled_indices] + ["Y"]

        return shuffled_data, new_col_name
    
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


def add_true_outliers(data, outlier_percentage, std_multiplier=3):
    n_outliers = int(len(data) * outlier_percentage / 100)
    outlier_indices = np.random.choice(len(data), n_outliers, replace=False)

    data_array = data.values.copy()
    
    n_features = data_array.shape[1] - 1

    # Compute mean and std for Y
    mean_Y = data.iloc[:, -1].mean()
    std_Y = data.iloc[:, -1].std()

    for idx in outlier_indices:
        for col in range(n_features):
            mean = np.mean(data_array[:, col])
            std = np.std(data_array[:, col])
            direction = np.random.choice([-1, 1])
            data_array[idx, col] = mean + (direction * std_multiplier * std)
        
        # Modify Y 
        direction_Y = np.random.choice([-1, 1])
        data_array[idx, -1] = mean_Y + (direction_Y * std_multiplier * std_Y)

    col_names = [f"X{i}" for i in range(n_features)] + ["Y"]
    result_df = pd.DataFrame(data_array, columns=col_names)
    
    return result_df


def add_outliers(data, outlier_percentage, percentile_threshold=95):
    n_outliers = int(len(data) * outlier_percentage / 100)
    outlier_indices = np.random.choice(len(data), n_outliers, replace=False)

    data_array = data.values.copy()
    
    n_features = data_array.shape[1] - 1

    # Compute percentile for X and Y
    lower_bounds_X = np.percentile(data_array[:, :-1], 100 - percentile_threshold, axis=0)
    upper_bounds_X = np.percentile(data_array[:, :-1], percentile_threshold, axis=0)
    lower_bound_Y = np.percentile(data_array[:, -1], 100 - percentile_threshold)
    upper_bound_Y = np.percentile(data_array[:, -1], percentile_threshold)

    for idx in outlier_indices:
        feature_directions = np.random.choice([-1, 1], size=n_features)  
        for col in range(n_features):
            if feature_directions[col] == -1:
                data_array[idx, col] = np.random.uniform(lower_bounds_X[col] - abs(lower_bounds_X[col] * 0.2), lower_bounds_X[col])
            else:
                data_array[idx, col] = np.random.uniform(upper_bounds_X[col], upper_bounds_X[col] + abs(upper_bounds_X[col] * 0.2))
    
    Y_directions = np.random.choice([-1, 1], size=n_outliers)  
    Y_outliers = np.where(
        Y_directions == -1, 
        np.random.uniform(lower_bound_Y - abs(lower_bound_Y * 0.2), lower_bound_Y, size=n_outliers),
        np.random.uniform(upper_bound_Y, upper_bound_Y + abs(upper_bound_Y * 0.2), size=n_outliers)
    )
    data_array[outlier_indices, -1] = Y_outliers  # Apply the Y outlier values

    col_names = [f"X{i}" for i in range(n_features)] + ["Y"]
    result_df = pd.DataFrame(data_array, columns=col_names)
    
    return result_df


def format_decimal(x, digits):
    formatted_value = f"{x:.{digits}f}"
    return float(formatted_value)

def generate_combination(n,p, seed=12345):
    dgps = ['linear']
    #dgps = ['sigmoid2']
    distributions = ['exp']
    #sign_controls = ['all_positive_large_scale']
    sign_controls = 'all_positive2'
    #scales = 'fixed_scale'
    #outliers = ['fixed_outlier']
    #digits = ['fixed_digit', 'random_digit']
    #digits = '14_digits'
    #outlier_percentages = [0.5, 1, 5]
    #column_order = 'shuffle'
    #column_name = 'diff_name4'

    np.random.seed(seed)
    beta = np.random.uniform(0, 1, p)
    print(beta)
    # [0.92961609 0.31637555 0.18391881 0.20456028 0.56772503 0.5955447 0.96451452 0.6531771  0.74890664 0.65356987]
    #outlier_values = generate_outliers(4,n)
    
    for dgp in dgps:
        for distribution in distributions:
                data, col_names = simulate_data(n,p,dgp=dgp, distribution=distribution, sign_control=sign_controls,scale=None, outlier_values= None, digit=None, beta=beta, column_order=None, column_name=column_name）

                '''
                for outlier_pct in outlier_percentages:
                    data_with_outliers = add_outliers(data, outlier_pct)
                    outlier_filename = f"data/synthetic_data_{dgp}_{distribution}_{sign_control}_Realoutliers_{outlier_pct}.csv"
                    data_with_outliers.to_csv(outlier_filename, index=False)
                '''
                file_name = f"data/synthetic_data_{dgp}_{distribution}_{sign_controls}_{column_name}.csv"
                print(file_name)
                data.to_csv(file_name, header=col_names, index=False)
                print(f"Generated and saved: {file_name}")



if __name__ == "__main__":
    generate_combination(5000, 10)
