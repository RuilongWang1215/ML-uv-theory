import os
import pandas as pd
import numpy as np
import json
import time
from dso import DeepSymbolicOptimizer
import psutil

# Set the path
dir_path = os.path.dirname(__file__)
data_path = os.path.join(dir_path, 'data', 'processed_data')

if not os.path.exists(dir_path+'/result_dso'):
    os.makedirs(dir_path+'/result_dso')
if not os.path.exists(dir_path+'/result_dso/figures'):
    os.makedirs(dir_path+'/result_dso/figures')
if not os.path.exists(dir_path+'/data/dso_config'):
    os.makedirs(dir_path+'/data/dso_config')

config_path = os.path.join(dir_path, 'data','dso_config')
result_path = os.path.join(dir_path, 'result_dso')

class dso_config:
    def __init__(self, file_name, function_set, n_samples = 1000000, batch_size = 5000):
        self.file_name = file_name
        self.data_path = os.path.join(data_path, f'{file_name}.csv')
        self.core = psutil.cpu_count(logical=False)
        self.config = {
            "experiment" : {
                "logdir" : result_path,
                "exp_name" : self.file_name+time.strftime("_%m-%d_%H%M"),
                "seed" : 0
            },
            
            "task" : {
                "task_type" : "regression",
                "dataset" : self.data_path,
                "function_set":function_set,
                "metric" : "inv_nrmse",
                "metric_params" : [1.0],
                "threshold" : 1e-12,
                "poly_optimizer_params" : {
                    "degree": 5,
                    "coef_tol": 1e-6,
                    "regressor": "dso_least_squares",}
            },
            
            "training" : {
                "n_samples" : n_samples,
                "batch_size" : batch_size,
                "epsilon" : 0.05,
                "baseline" : "R_e",
                "alpha" : 0.5,
                "b_jumpstart" : False,
                "n_cores_batch" : self.core,
                "complexity" : "token",
                "const_optimizer" : "scipy",
                "const_params" : {
                    "method" : "L-BFGS-B",
                    "options" : {
                        "gtol" : 1e-3
                    }
                },
            },

            "logging" : {
                "save_all_iterations" : False,
                "save_summary" : False,
                "save_positional_entropy" : False,
                "save_pareto_front" : True,
                "save_cache" : False,
                "save_cache_r_min" : 0.9,
                "save_freq" : 1,
                "save_token_count" : False,
                "hof" : 100
            },

            "state_manager": {
                "type" : "hierarchical",
                "observe_action" : False,
                "observe_parent" : True,
                "observe_sibling" : True,
                "observe_dangling" : False,
                "embedding" : False,
                "embedding_size" : 8
            },

            "policy" : {
                "policy_type" : "rnn", 
                "max_length" : 64,
                "cell" : "lstm",
                "num_layers" : 1,
                "num_units" : 32,
                "initializer" : "zeros"
            },

            "policy_optimizer" : {
                "policy_optimizer_type" : "pg", 
                "summary" : False,
                "learning_rate" : 0.001,
                "optimizer" : "adam",
                "entropy_weight" : 0.005,
                "entropy_gamma" : 1.0
            },

            "gp_meld" : {
                "run_gp_meld" : False,
                "verbose" : False,
                "generations" : 20,
                "p_crossover" : 0.5,
                "p_mutate" : 0.5,
                "tournament_size" : 5,
                "train_n" : 50,
                "mutate_tree_max" : 3,
                "parallel_eval" : False
            },

            "prior": {
                "count_constraints" : False,
                "relational" : {
                    "targets" : [],
                    "effectors" : [],
                    "relationship" : None,
                    "on" : False
                },

                "length" : {
                    "min_" : 4,
                    "max_" : 35,
                    "on" : False
                },

                "repeat" : {
                    "tokens" : "const",
                    "min_" : None,
                    "max_" : 5,
                    "on" : False
                },
                "inverse" : {"on" : False},
                "trig" : {"on" : False},
                "const" : {"on" : False},
                "no_inputs" : {"on" : False},
                "uniform_arity" : {"on" : False},
                "soft_length" : {
                    "loc" : 10,
                    "scale" : 5,
                    "on" : False
                },
                "domain_range" : {"on" : False},
                "language_model" : {
                    "weight" : None,
                    "on" : False
                },
                "multi_discrete" : {
                    "dense" : False,           
                    "ordered" : False,        
                    "on" : False
                }
            },

            "postprocess" : {
                "show_count" : 5,
                "save_plots" : True
            }
            }
        
    def to_json(self):
        filepath=os.path.join(config_path, self.file_name+'_config.json')
        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=3)
        print(f"Config saved to {filepath}")
    
    def run_dso(self):
        model = DeepSymbolicOptimizer(os.path.join(config_path, self.file_name+'_config.json'))
        model.train()
        print(f"Model for {self.file_name} has been saved to {result_path}")




