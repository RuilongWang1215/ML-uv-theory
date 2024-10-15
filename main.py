import os
import time
import pandas as pd
import numpy as np
from SR_PYSR import *

###### Put the settings here ######
SUBSTANCE = 'combined_data_add_features_3_filtered_normalized'
TEST_RATIO = 0.2
ITERATION = 500
MAXSIZE =30

###### Load the data ######
script_dir = os.path.dirname(__file__)
data_dir = os.path.join(script_dir, 'data', 'processed_data')
data = pd.read_csv(os.path.join(data_dir, f'{SUBSTANCE}.csv'))
X = data.drop(columns=['delta_phi'])
y = data['delta_phi']

##### Begin the symbolic regression ######
ts = time.time()
pysr = PYSR_wrapper(substance=SUBSTANCE, X=X, y=y, test_ratio=TEST_RATIO, 
                    iteration =ITERATION, MAXSIZE= MAXSIZE)
pysr.run_SR()
te = time.time()
print(f'Time taken: {round((te-ts)/60, 2)} minutes')