import os
import time
import pandas as pd
import numpy as np
from data_preprocessing import data_preprocess
#from data_preprocessing import data_preprocess

# CREATED: 2024-10-21
# AUTHOR: Ruilong Wang
# PURPOSE: This script is used to run symbolic regression using DSO and PYSR
# INPUT: file name, test ratio, iteration, maxsize
# OUTPUT: symbolic regression model, figures, and csv files

###### Put the settings here ######
FILE = 'all_add_features_manual_normalized_n'
TEST_RATIO = 0.2
ITERATION = 500
MAXSIZE =35
Algorithm = 'PYSR'   # 'DSO' or 'PYSR'
Loadingfrom = 'file' # 'file' or 'data_preprocess'
###### Load the data ######

# Load data from file
if Loadingfrom == 'file':
    script_dir = os.path.dirname(__file__)
    data_dir = os.path.join(script_dir, 'data', 'processed_data')
    data = pd.read_csv(os.path.join(data_dir, f'{FILE}.csv'))
    data = data.dropna(axis=1)
    X = data.drop(columns=['delta_phi'])
    y = data['delta_phi']
    print(f"length of X: {len(X)}") 

# Load data from data_preprocess
if Loadingfrom == 'data_preprocess':
    dp = data_preprocess()
    supplement_features= dp.feature_engineering_manual()
    data = dp.combing_data(mode = 'all', sample_ratio=0.7)
    data = dp.add_features(data, supplement_features)
    data = dp.compute_dimensionless(data)
    data = dp.feature_selection(data)
    data =data.dropna(axis=1)
    X = data.drop(columns=['delta_phi'])
    y = data['delta_phi']
    print(f"length of X: {len(X)}")

### Begin DSO symbolic regression ######
if Algorithm == 'DSO':
    from SR_DSO import *
    ts = time.time()
    dataset = (X, y)
    Function_Set=["add", "sub", "mul", "div", "sin", "cos", "exp", "log", "const","tanh",
                "inv",]
    dso = dso_config(file_name=FILE, function_set=Function_Set, n_samples = 2700000, batch_size = 18000)
    dso.to_json()
    dso.run_dso()
    te = time.time()
    print(f'Time taken: {round((te-ts)/3600, 2)} hours')



##### Begin the symbolic regression ######
if Algorithm == 'PYSR':
    from SR_PYSR import *
    # run the SR for 5 times
    run_time = 5
    while run_time > 0:
        ts = time.time()
        pysr = PYSR_wrapper(substance=FILE, X=X, y=y, test_ratio=TEST_RATIO, 
                            iteration =ITERATION, MAXSIZE= MAXSIZE)
        fulfillment = pysr.run_SR()
        te = time.time()
        print(f'Time taken: {round((te-ts)/60, 2)} minutes')
        if fulfillment == True:
            run_time =run_time -1
        else:
            run_time = run_time