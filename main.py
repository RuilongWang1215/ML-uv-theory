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
#FILE = 'all_add_features_manual_normalized_n'
TEST_RATIO = 0.2
ITERATION = 500
MAXSIZE =35
Algorithm = 'PYSR'   # 'DSO' or 'PYSR'
Loadingfrom = 'file' # 'file' or 'data_preprocess'
#stepwise = True
###### Load the data ######
FILES = [#'all_add_features_manual_normalized_n', 
         #'all_add_features_manual_normalized_n', 
         #'all_add_features_manual_normalized_n', ]
         'all_add_features_manual_normalized_n_filtered']
stepwises= [#False, 
            #True ]
            False]
WEIGHTEDs = [#True, 
             #False, 
             False]
for FILE, stepwise, WEIGHTED in zip(FILES, stepwises, WEIGHTEDs):
    run_time = 3
    print(f"FILE: {FILE}")
    print(f"stepwise: {stepwise}")
    print(f"WEIGHTED: {WEIGHTED}")

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
        
        if not stepwise:
            while run_time > 0:
                ts = time.time()
                pysr = PYSR_wrapper(substance=FILE, X=X, y=y, test_ratio=TEST_RATIO, 
                                    iteration =ITERATION, MAXSIZE= MAXSIZE, WEIGHTED=WEIGHTED)
                fulfillment, model = pysr.run_SR()
                te = time.time()
                print(f'Time taken: {round((te-ts)/60, 2)} minutes')
                if fulfillment == True:
                    run_time =run_time -1
                else:
                    run_time = run_time
                    
        if stepwise:
            ts = time.time()
            nu_value, count = np.unique(data['nu'], return_counts=True)
            nu_count = dict(zip(nu_value, count))
            nu_count = dict(sorted(nu_count.items(), key=lambda x: x[1], reverse=False))
            nu_value = list(nu_count.keys())
            count = list(nu_count.values())
            
            data_first = data[data['nu'] != nu_value[0]]
            data_second = data[data['nu'] == nu_value[0]]
            print(f"length of data_first: {len(data_first)}", f"length of data_second: {len(data_second)}")
            X_first = data_first.drop(columns=['delta_phi'])
            y_first = data_first['delta_phi']
            X_second = data_second.drop(columns=['delta_phi'])
            y_second_0 = data_second['delta_phi']
            save_path = os.path.join(script_dir, 'result_pysr')
            while run_time > 0:
                FILE_name_1 = FILE + '_'+str(run_time)+'_a'
                pysr_1 = PYSR_wrapper(substance=FILE_name_1, X=X_first, y=y_first, test_ratio=TEST_RATIO, 
                                    iteration =ITERATION, MAXSIZE= MAXSIZE)
                fulfillment_1, model_1 = pysr_1.run_SR()
                fulfillment_2 = False
                while fulfillment_1 and not fulfillment_2:
                    y_second = model_1.predict(X_second)-y_second_0
                    FILE_name_2 = FILE + '_'+str(run_time)+'_w'
                    pysr_2 = PYSR_wrapper(substance=FILE_name_2, X=X_second, y=y_second, test_ratio=TEST_RATIO, 
                                    iteration =ITERATION, MAXSIZE= MAXSIZE)
                    fulfillment_2, model_2 = pysr_2.run_SR()
                    if fulfillment_2:
                        run_time =run_time -1
                        break
                    else:
                        run_time = run_time
                    
            
            te = time.time()
            print(f'Time taken: {round((te-ts)/60, 2)} minutes')
        
