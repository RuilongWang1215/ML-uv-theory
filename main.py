import os
import time
import pandas as pd
import numpy as np
from SR_PYSR import *


SUBSTANCE = 'methane'
TEST_RATIO = 0.2
ITERATION = 100
MAXSIZE =30
script_dir = os.path.dirname(__file__)
data_dir = os.path.join(script_dir, 'data')
data = pd.read_csv(os.path.join(data_dir, f'real_ufrac_{SUBSTANCE}.csv'))
X = data[['density', 'temperature']]
y = data['delta_phi']

ts = time.time()
pysr = PYSR_wrapper(substance=SUBSTANCE, X=X, y=y, test_ratio=TEST_RATIO, 
                    iteration =ITERATION, MAXSIZE= MAXSIZE)
pysr.run_SR()
te = time.time()
print(f'Time taken: {round((te-ts)/60, 2)} minutes')