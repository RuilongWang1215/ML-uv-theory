# This script is the symbolic regression framework for modelling the correction term of uv-theory
import numpy as np
import pandas as pd
import time
from pysr import PySRRegressor
import sympy as sp
# import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import os
import psutil
import numpy as np


# This function is used to organize the files generated by the symbolic regression
'''It will move the generated files to the 'result' folder and rename them
The naming convention is 'uv_trial_{substance}_{count}.csv'
The count is the number of files with the same substance name'''

class PYSR_wrapper():
    def __init__(self, substance, X, y, test_ratio=0.2, iteration=100, MAXSIZE=35):
        self.base_path = os.path.dirname(__file__)
        self.substance = substance
        self.model = None
        self.NAME = 'pySR_' + substance+ '_iter'+ str(iteration)+ time.strftime("_%H%M")
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.test_ratio = test_ratio
        self.iteration = iteration
        self.maxsize = MAXSIZE
        self.X = X
        self.y = y
        
    def organize_files(self):
        if not os.path.exists(self.base_path+'/result_pysr'):
            os.makedirs(self.base_path+'/result_pysr')
        file_names = os.listdir()
        file_in_result = os.listdir(self.base_path+'/result_pysr')
        counts = {}
        for file in file_in_result:
            if file.startswith(self.NAME):
                counts[file] = file.split('_')[-1].split('.')[0]
        if counts:
            # delete counts value larger than 1000
            counts = {k: v for k, v in counts.items() if int(v) < 1000}
            if counts:        
                max_count = max([int(count) for count in counts.values()])
            else:
                max_count = 0
            self.NAME = self.NAME + '_' + str(max_count + 1)

        for file_name in file_names:
            if file_name.startswith('hall_of_fame'):
                if file_name.endswith('.csv'):
                    os.replace(file_name, self.base_path+'/result_pysr/' + self.NAME + '.csv')
                if file_name.endswith('.pkl'):
                    os.replace(file_name, self.base_path+'/result_pysr/' + self.NAME +'.pkl')
                if file_name.endswith('.bkup'):
                    os.remove(file_name)
                    
    def X_transform(self, X):
        X_transformed = X.copy()  
        X_transformed.loc[:,'density'] = X_transformed.loc[:,'density'] / (1 + X_transformed.loc[:,'density']**2)
        return X_transformed

    def data_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X, self.y, test_size=self.test_ratio)
        
    def plot_regression(self):
        y_pred = self.model.predict(self.X_test)
        r2 = r2_score(self.y_test, y_pred)
        fig = plt.figure()
        plt.plot(self.y_test, y_pred, 'o', color='skyblue')
        plt.plot(self.y_test, self.y_test, 'slategrey')
        plt.text(0.7, 0.2, f'R2: {r2:.2f}', transform=fig.transFigure)
        plt.xlabel('True')
        plt.ylabel('Predicted')
        #plt.title(f"True vs Predicted {self.substance} Correction Term")
        if not os.path.exists(self.base_path+'/result_pysr/pictures'):
            os.makedirs(self.base_path+'/result_pysr/pictures')
        plt.savefig(self.base_path+f"/result_pysr/pictures/{self.NAME}_{round(r2,4)}.png", dpi=300, bbox_inches='tight')
    
    def run_SR(self):
        self.data_split()
        density = self.X_train['density']
        weights = density/(10**(-16)+density**1.005)
        """custom_loss = 
        loss(prediction, target, X) = (prediction - target)^2 
        + 100 * (prediction - 0)^2 * (abs(X[:, 0]) < 1e-6) 
        + 100 * (prediction - 0)^2 * (abs(X[:, 0]) > 1e6) 
        """
        model = PySRRegressor(
            niterations=self.iteration,
            procs=psutil.cpu_count(),  # use all available cores
            populations=psutil.cpu_count()*3,
            maxsize = self.maxsize,
            binary_operators=["+", "*", "-", "/"],
            unary_operators=[
                "cos",
                "sin",
                "exp",
                "tan",
                #"inv(x) = 1/x",
                "sqrt",
                "log",
                "square",
                "cube",
                "sinh(x) = (exp(x) - exp(-x))/2",
                "cosh(x) = (exp(x) + exp(-x))/2",
            ],
            extra_sympy_mappings={#"inv": lambda x: 1 / x,
                                "sinh": lambda x: (sp.exp(x) - sp.exp(-x)) / 2,
                                "cosh": lambda x: (sp.exp(x) + sp.exp(-x)) / 2,
                                },
            # ^ Define operator for SymPy as well
            weights=weights,
            elementwise_loss="loss(prediction, target, w) = (w*prediction - target)^2",
            ncycles_per_iteration = self.iteration,
            
            timeout_in_seconds = 60*60*8,
            model_selection='accuracy',
            progress= True,
            batching = True,
            bumper = True,)  
        model.fit(self.X_train, self.y_train, weights=weights)
        self.model = model
        self.plot_regression()
        self.organize_files()
        print(f"Model for {self.substance} has been saved")
        return model




