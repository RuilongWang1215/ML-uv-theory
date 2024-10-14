import os
from SR_PYSR import *


print("Loading data...")    
script_dir = os.path.dirname(__file__)
data_dir = os.path.join(script_dir, 'data')
data = pd.read_csv(os.path.join(data_dir, 'real_ufrac_argon.csv'))
X = data[['density', 'temperature']]
y = data['delta_phi']

pysr = PYSR_wrapper(substance='argon', X=X, y=y, test_ratio=0.2, iteration =100, MAXSIZE= 30)
pysr.run_SR()