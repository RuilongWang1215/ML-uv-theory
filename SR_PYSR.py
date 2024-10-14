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
        self.substance = substance
        self.model = None
        self.X = X
        self.y = y
        self.NAME = 'pySR_' + substance+ '_'+ time.strftime("%Y%m%d")
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.test_ratio = test_ratio
        self.iteration = iteration
        self.maxsize = MAXSIZE
        
    def organize_files(self):
        file_names = os.listdir()
        file_in_result = os.listdir('result')
        counts = {}
        for file in file_in_result:
            if file.startswith(self.NAME):
                counts[file] = file.split('_')[-1].split('.')[0]
        if counts:
            max_count = max([int(count) for count in counts.values()])
            self.NAME = self.NAME + '_' + str(max_count + 1)

        for file_name in file_names:
            if file_name.startswith('hall_of_fame'):
                if file_name.endswith('.csv'):
                    os.replace(file_name, 'result/' + self.NAME + '.csv')
                if file_name.endswith('.pkl'):
                    os.replace(file_name, 'result/' + self.NAME +'.pkl')
                if file_name.endswith('.bkup'):
                    os.remove(file_name)

    def data_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X, self.y, test_size=self.test_ratio, random_state=42)
        
    def plot_regression(self):
        y_pred = self.model.predict(self.X_test)
        r2 = r2_score(self.y_test, y_pred)
        fig = plt.figure()
        plt.plot(self.y_test, y_pred, 'o', color='skyblue')
        plt.plot(self.y_test, self.y_test, 'slategrey')
        plt.text(0.7, 0.2, f'R2: {r2:.2f}', transform=fig.transFigure)
        plt.xlabel('True')
        plt.ylabel('Predicted')
        plt.title(f"True vs Predicted {self.substance} Correction Term")
        if not os.path.exists('result/pictures'):
            os.makedirs('result/pictures')
        plt.savefig(f"result/pictures/{self.NAME}_{round(r2,4)}.png", dpi=300, bbox_inches='tight')
    
    def run_SR(self):
        if self.substance == 'argon':
            BATCHING = True
        else:
            BATCHING = False
        
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
                "inv(x) = 1/x",
                "sqrt",
                "log",
                "square",
                "cube",
                "sinh(x) = (exp(x) - exp(-x))/2",
                "cosh(x) = (exp(x) + exp(-x))/2",
                # ^ Custom operator (julia syntax)
            ],
            extra_sympy_mappings={"inv": lambda x: 1 / x,
                                    "sinh": lambda x: (sp.exp(x) - sp.exp(-x)) / 2,
                                        "cosh": lambda x: (sp.exp(x) + sp.exp(-x)) / 2},
            # ^ Define operator for SymPy as well
            elementwise_loss="loss(prediction, target) = (prediction - target)^2",
            ncycles_per_iteration = self.iteration,
            timeout_in_seconds = 60*60*8,
            progress= True,
            batching = BATCHING,
            bumper = True,)  
        self.data_split()
        model.fit(self.X_train, self.y_train)
        self.model = model
        self.plot_regression()
        self.organize_files()
        print(f"Model for {self.substance} has been saved")
        return model




