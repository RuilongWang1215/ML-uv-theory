import os
import time
import pandas as pd
import numpy as np
from data_preprocessing import data_preprocess

# CREATED: 2024-10-21
# AUTHOR: Ruilong Wang
# PURPOSE: Run symbolic regression using DSO and PYSR

###### Configuration ######
TEST_RATIO = 0.2
ITERATION = 500
MAXSIZE = 35
RUN_TIME = 10
ALGORITHM = 'PYSR'  # Options: 'DSO' or 'PYSR'
LOADING_FROM = 'file'  # Options: 'file' or 'data_preprocess'

FILES = [
    'all_add_features_manual_normalized_n',
    'all_add_features_manual_normalized_n',
    'all_add_features_manual_normalized_n',
    'all_add_features_manual_normalized_n_filtered'
]
STEPWISE_FLAGS = [False, False, True, False]
WEIGHTED_FLAGS = [False, True, False, False]

###### Functions ######
def load_data(file_name, loading_from, data_dir=None):
    """Load and preprocess data based on the source."""
    if loading_from == 'file':
        if data_dir is None:
            raise ValueError("Data directory must be specified for file loading.")
        data = pd.read_csv(os.path.join(data_dir, f"{file_name}.csv")).dropna(axis=1)
    elif loading_from == 'data_preprocess':
        dp = data_preprocess()
        supplement_features = dp.feature_engineering_manual()
        data = dp.combing_data(mode='all', sample_ratio=0.7)
        data = dp.add_features(data, supplement_features)
        data = dp.compute_dimensionless(data)
        data = dp.feature_selection(data).dropna(axis=1)
    else:
        raise ValueError("Invalid LOADING_FROM option.")
    
    X = data.drop(columns=['delta_phi'])
    y = data['delta_phi']
    return data, X, y

def run_dso(file_name, X, y):
    """Run symbolic regression using DSO."""
    from SR_DSO import dso_config
    ts = time.time()
    function_set = [
        "add", "sub", "mul", "div", "sin", "cos", "exp", "log",
        "const", "tanh", "inv"
    ]
    dso = dso_config(
        file_name=file_name, function_set=function_set,
        n_samples=2700000, batch_size=18000
    )
    dso.to_json()
    dso.run_dso()
    te = time.time()
    print(f"Time taken: {round((te - ts) / 3600, 2)} hours")

def run_pysr(file_name, data, X, y, stepwise, weighted):
    """Run symbolic regression using PYSR."""
    from SR_PYSR import PYSR_wrapper
    run_time = RUN_TIME

    if not stepwise:
        while run_time > 0:
            ts = time.time()
            pysr = PYSR_wrapper(
                substance=file_name, X=X, y=y,
                test_ratio=TEST_RATIO, iteration=ITERATION,
                MAXSIZE=MAXSIZE, WEIGHTED=weighted
            )
            fulfillment, _ = pysr.run_SR()
            te = time.time()
            print(f"Time taken: {round((te - ts) / 60, 2)} minutes")
            run_time -= 1 if fulfillment else 0
    else:
        # Stepwise logic
        ts = time.time()
        nu_values, counts = np.unique(data['nu'], return_counts=True)
        nu_count = dict(zip(nu_values, counts))
        nu_count = dict(sorted(nu_count.items(), key=lambda x: x[1]))

        data_first = data[data['nu'] != list(nu_count.keys())[0]]
        data_second = data[data['nu'] == list(nu_count.keys())[0]]
        print(f"Length of data_first: {len(data_first)}, data_second: {len(data_second)}")

        X_first = data_first.drop(columns=['delta_phi'])
        y_first = data_first['delta_phi']
        X_second = data_second.drop(columns=['delta_phi'])
        y_second_0 = data_second['delta_phi']

        while run_time > 0:
            # Step 1: Train on data_first
            file_name_1 = f"{file_name}_{run_time}_a"
            pysr_1 = PYSR_wrapper(
                substance=file_name_1, X=X_first, y=y_first,
                test_ratio=TEST_RATIO, iteration=ITERATION, MAXSIZE=MAXSIZE
            )
            fulfillment_1, model_1 = pysr_1.run_SR()

            if not fulfillment_1:
                break

            # Step 2: Adjust residuals for data_second
            y_second = y_second_0 - model_1.predict(X_second)

            # Step 3: Train on residuals
            file_name_2 = f"{file_name}_{run_time}_w"
            pysr_2 = PYSR_wrapper(
                substance=file_name_2, X=X_second, y=y_second,
                test_ratio=TEST_RATIO, iteration=ITERATION, MAXSIZE=10
            )
            fulfillment_2, _ = pysr_2.run_SR()

            if fulfillment_2:
                run_time -= 1
            else:
                break
        te = time.time()
        print(f"Time taken: {round((te - ts) / 60, 2)} minutes")

###### Main Script ######
if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    data_dir = os.path.join(script_dir, 'data', 'processed_data')

    for file_name, stepwise, weighted in zip(FILES, STEPWISE_FLAGS, WEIGHTED_FLAGS):
        print(f"Processing FILE: {file_name}, stepwise: {stepwise}, WEIGHTED: {weighted}")

        # Load data
        data, X, y = load_data(file_name, LOADING_FROM, data_dir)

        print(f"Length of X: {len(X)}")

        # Run the appropriate algorithm
        if ALGORITHM == 'DSO':
            run_dso(file_name, X, y)
        elif ALGORITHM == 'PYSR':
            run_pysr(file_name, data, X, y, stepwise, weighted)
        else:
            raise ValueError("Invalid ALGORITHM option.")

