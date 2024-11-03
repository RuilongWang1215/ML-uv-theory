# this is the script to evaluate if the function fulfill the requirements
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from sympy import symbols, limit, oo
from pysr import PySRRegressor
import signal

# Define a custom exception for the timeout
class TimeoutException(Exception):
    pass

# Define the timeout handler
def timeout_handler(signum, frame):
    raise TimeoutException("Function call timed out")

# Define the timeout decorator
def timeout_decorator(seconds=300):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Set the signal handler and a timeout alarm
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                # Disable the alarm
                signal.alarm(0)
            return result
        return wrapper
    return decorator

# Modify the restriction_check method to use the decorator
@timeout_decorator(seconds=300)  # 300 seconds = 5 minutes
def run_restriction_check(restriction_instance):
    return restriction_instance.restriction_check()

class Restriction:
    def __init__(self, model):
        self.model = model
        self.equation = model.sympy()  # Removed simplify for speed
        self.rho, self.T = symbols('density temperature')
        self.complexity = model.get_best()['complexity']
        
    def helmholtz_energy_restriction(self):
        # Restriction 1: lim_{ρ -> 0} (δφ * ρ) = 0
        equal_1 = self.equation * self.rho
        restriction_1 = equal_1.subs(self.rho, 0) == 0
        # Restriction 2: lim_{T -> ∞} δφ is not infinite (not equal to oo or -oo)
        limit_value = limit(self.equation, self.T, oo)
        # or is_finite
        restriction_2 = limit_value != oo and limit_value != -oo
        return restriction_1, restriction_2
    
    def pressure_restriction(self):
        # Restriction 1: lim_{ρ -> 0} ρ^3 * (d(δφ) / dρ) = 0
        phi_rho = self.equation.diff(self.rho)
        restriction1 = (self.rho**3 * phi_rho).subs(self.rho, 0) == 0

        # Restriction 2: lim_{T -> ∞} (d(δφ * ρ) / dρ) is not infinite
        phi_rho_rho = (self.equation * self.rho).diff(self.rho)
        limit_value = limit(phi_rho_rho, self.T, oo)
        restriction2 = limit_value != oo and limit_value != -oo
        return restriction1, restriction2
    
    def internal_energy_restriction(self):
        # Restriction 1: lim_{ρ -> 0} ρ * (d(δφ) / dT) = 0
        phi_T = self.equation.diff(self.T)
        restriction1 = (self.rho * phi_T).subs(self.rho, 0) == 0

        # Restriction 2: lim_{T -> ∞} (d(δφ) / dT) is not infinite
        limit_value = limit(phi_T, self.T, oo)
        restriction2 = limit_value != oo and limit_value != -oo
        return restriction1, restriction2
    
    def isochoric_heat_capacity_restriction(self):
        # Restriction: lim_{ρ -> 0} ρ * (d^2(δφ) / dT^2) = 0
        phi_TT = self.equation.diff(self.T, 2)
        restriction = (self.rho * phi_TT).subs(self.rho, 0) == 0
        return restriction
    
    def restriction_check(self):
        print(f"Evaluating restrictions for {self.equation}")
        try:
            helmholtz = self.helmholtz_energy_restriction()
            pressure = self.pressure_restriction()
            internal_energy = self.internal_energy_restriction()
            heat_capacity = self.isochoric_heat_capacity_restriction()
            
            restrictions = {
                "Helmholtz Energy": helmholtz,
                "Pressure": pressure,
                "Internal Energy": internal_energy,
                "Isochoric Heat Capacity": heat_capacity
            }
            
            # Check and print results
            for name, result in restrictions.items():
                print(f"{name} restrictions:")
                if isinstance(result, tuple):  # Some have two restrictions
                    for idx, res in enumerate(result, start=1):
                        print(f"  Restriction {idx}: {'Satisfied' if res else 'Not satisfied'}")
                else:
                    print(f"  Restriction: {'Satisfied' if result else 'Not satisfied'}")
        except:
            restrictions = {
                "Helmholtz Energy": False,
                "Pressure": False,
                "Internal Energy": False,
                "Isochoric Heat Capacity": False
            }
            print("An error occurred while evaluating the restrictions.")
        return restrictions

if __name__ == '__main__':
    ts = time.time()
    base_path = os.path.dirname(__file__)
    #results = pd.read_csv(f'{base_path}/result_pysr/restriction_results.csv')
    #calculated = results['Filename'].values
    SUBSTANCES = os.listdir(f'{base_path}/result_pysr')
    # SUBSTANCES is the name that before . and begin with pySR
    SUBSTANCES = [substance.split('.')[0] for substance in SUBSTANCES if substance.endswith('.pkl')]
    #SUBSTANCES = [substance for substance in SUBSTANCES if substance not in calculated]
    fulfills = {}
    equations = []
    complexitys = []
    for idx, substance in enumerate(SUBSTANCES):
        print(f"Checking restrictions for {substance} ({idx}/{len(SUBSTANCES)})")
        model = PySRRegressor.from_file(f'{base_path}/result_pysr/{substance}.pkl')
        r = Restriction(model)
        equations.append(r.equation)
        complexitys.append(r.complexity)
        
        try:
            # Run restriction_check with a timeout
            restrictions_result = run_restriction_check(r)
            
            # Check if all restrictions are satisfied
            if all(all(res) if isinstance(res, (list, tuple)) else res for res in restrictions_result.values()):
                fulfills[substance] = True
            else:
                fulfills[substance] = False
        
        except TimeoutException:
            print(f"Timeout reached for {substance}. Moving to the next iteration.")
            fulfills[substance] = False  # Mark as not fulfilling restrictions due to timeout

    # Output results
    print(f"Fulfills restrictions: {fulfills}")
    # Save the results to a CSV file
    df = pd.DataFrame({
        "Filename": SUBSTANCES,
        "Equation": equations,
        "Complexity": complexitys,
        "Fulfillment": [fulfills.get(substance, False) for substance in SUBSTANCES]
    })
    #df = pd.concat([results, df], ignore_index=True, axis=0)
    df.to_csv(f'{base_path}/result_pysr/restriction_results.csv', index=False)
    print("Done!")


