import os
import pandas as pd
import numpy as np
import json
from dso import DeepSymbolicOptimizer
import psutil

# Set the path
base_path = os.path.abspath(__file__)
dir_path = os.path.dirname(base_path)
data_path = os.path.join(dir_path, 'data', 'processed_data')

if not os.path.exists(dir_path+'/result_dso'):
    os.makedirs(dir_path+'/result_dso')
if not os.path.exists(dir_path+'/result_dso/figures'):
    os.makedirs(dir_path+'/result_dso/figures')
if not os.path.exists(dir_path+'data/dso_config'):
    os.makedirs(dir_path+'data/dso_config')

config_path = os.path.join(dir_path, 'data', 'dso_config')
result_path = os.path.join(dir_path, 'result_dso')

function_set=["add", "sub", "mul", "div", "sin", "cos", "exp", "log", "const",]
class dso_config:
    def __init__(self, file_name, function_set):
        self.file_name = file_name
        self.dataset_path = os.path.join(data_path, file_name)+'.csv'
        self.core = psutil.cpu_count(logical=False)
        self.config = {
            "experiment" : {
                "logdir" : result_path,
                "exp_name" : null,
                "seed" : 0
            },
            
            "task" : {
                "task_type" : "regression",
                "dataset" : self.dataset_path,
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
                "n_samples" : 2000000,
                "batch_size" : 1000,
                "epsilon" : 0.05,
                "baseline" : "R_e",
                "alpha" : 0.5,
                "b_jumpstart" : false,
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
                "save_all_iterations" : false,
                "save_summary" : false,
                "save_positional_entropy" : false,
                "save_pareto_front" : true,
                "save_cache" : false,
                "save_cache_r_min" : 0.9,
                "save_freq" : 1,
                "save_token_count" : false,
                "hof" : 100
            },

            "state_manager": {
                "type" : "hierarchical",
                "observe_action" : false,
                "observe_parent" : true,
                "observe_sibling" : true,
                "observe_dangling" : false,
                "embedding" : false,
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
                "summary" : false,
                "learning_rate" : 0.001,
                "optimizer" : "adam",
                "entropy_weight" : 0.005,
                "entropy_gamma" : 1.0
            },

            "gp_meld" : {
                "run_gp_meld" : false,
                "verbose" : false,
                "generations" : 20,
                "p_crossover" : 0.5,
                "p_mutate" : 0.5,
                "tournament_size" : 5,
                "train_n" : 50,
                "mutate_tree_max" : 3,
                "parallel_eval" : false
            },

            "prior": {
                "count_constraints" : false,
                "relational" : {
                    "targets" : [],
                    "effectors" : [],
                    "relationship" : null,
                    "on" : false
                },

                "length" : {
                    "min_" : 4,
                    "max_" : 30,
                    "on" : false
                },

                "repeat" : {
                    "tokens" : "const",
                    "min_" : null,
                    "max_" : 3,
                    "on" : false
                },
                "inverse" : {"on" : false},
                "trig" : {"on" : false},
                "const" : {"on" : false},
                "no_inputs" : {"on" : false},
                "uniform_arity" : {"on" : false},
                "soft_length" : {
                    "loc" : 10,
                    "scale" : 5,
                    "on" : false
                },
                "domain_range" : {"on" : false},
                "language_model" : {
                    "weight" : null,
                    "on" : false
                },
                "multi_discrete" : {
                    "dense" : false,           
                    "ordered" : false,        
                    "on" : false
                }
            },

            "postprocess" : {
                "show_count" : 5,
                "save_plots" : true
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




