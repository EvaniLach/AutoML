#########################################################################################
# The assignment file. While it is allowed to change this entire file, we highly
# recommend using the provided template. YOU MUST USE THE RANGES AND HYPERPARAMETERS SPECIFIED
# IN GET_RANGES AND GET_CONFIG_PERFORMAMCE (IN SHORT: USE OUR SURROGATE PROBLEMS)
#########################################################################################

import numpy as np
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

from utils import GET_CONFIG_PERFORMANCE, GET_RANGES, SampleType, ParamType # make sure to make use of ParamType and SampleType in your code


parser = argparse.ArgumentParser()
parser.add_argument('--problem', choices=['good_range', 'bad_range', 'interactive'], required=True)
parser.add_argument('--algorithm', choices=['rs','tpe'], required=True)
parser.add_argument('--gamma', type=float, default=0.2)
parser.add_argument('--random_warmup', type=int, default=30)
parser.add_argument('--seed', type=int, default=42)
    

def random_search(problem, function_evaluations=150, **kwargs):
    """
    Function that performs random search on the given problem. It uses the 
    ranges and sampling types defined in GET_RANGES(problem) (see utils.py).

    Arguments:
      - problem (str): the problem identifier
      - function_evaluations (int): the number of configurations to evaluate
      - **kwargs: any other keyword arguments 

    Returns:
      - history (list): A list of the observed losses
      - configs (list): A list of the tried configurations. Every configuration is a dictionary
                        mapping hyperparameter names to the chosen values 
    """

    history = []
    configs = []

    # get all information about the hyperparameters we want to tune for this problem
    # (see utils.py) for the form of this.  
    RANGES = GET_RANGES(problem) 

    print(RANGES)

    # example of a hyperparameter configuration. you of course want to change the names and values
    # according to the problem and the values you sample
    config = {
        "hypername1": 5, 
        "hypername2": 10,
    }
    # once you have a configuration 'config' in the form of a dictionary mapping from 
    # hyperparameter -> value you can evaluate it using this function
    loss = GET_CONFIG_PERFORMANCE(config)

    print(loss)

    # TODO: implement the rest of the function
    return history, configs


def tpe(problem, function_evaluations=150, random_warmup=30, gamma=0.2, **kwargs):
    """
    Function that uses Tree Parzen Estimator (TPE) to tune the hyperparameters of the 
    given problem. It uses the ranges and sampling types defined in GET_RANGES(problem) 
    (see utils.py).

    Arguments:
      - problem (str): the prolbem identifier
      - function_evaluations (int): the number of configurations to evaluate
      - random_warmup (int): the number of initial iterations during which we perform random 
                             search
      - gamma: the value of gamma that determines the cutting point [good partition, bad partition]
      - **kwargs: any other keyword arguments 

    Returns:
      - history (list): A list of the observed losses
      - configs (list): A list of the tried configurations. Every configuration is a dictionary
                        mapping hyperparameter names to the chosen values 
    """

    history = []
    configs = []

    # get all information about the hyperparameters we want to tune for this problem
    # (see utils.py) for the form of this.  
    RANGES = GET_RANGES(problem) 

    # example of a hyperparameter configuration. you of course want to change the names and values
    # according to the problem and the values you sample
    config = {
        "hypername1": 5,
        "hypername2": 10,
    }
    # once you have a configuration 'config' in the form of a dictionary mapping from 
    # hyperparameter -> value you can evaluate it using this function
    loss = GET_CONFIG_PERFORMANCE(config)

    # TODO: implement the rest of the function
    return history, configs


###############################################################################################
# Code that parses command line arguments and saves the results
# code can be run by calling 
# python hyperopt.py --algorithm ALG_SPECIFIER --problem PROBLEM_SPECIFIER --more_arguments ...
# you do not need to change the code below
###############################################################################################
alg_fn = {'rs': random_search, 'tpe':tpe}
    
args = parser.parse_args()
np.random.seed(args.seed)

conf = vars(args)
tried_configs, performances = alg_fn[args.algorithm](**conf)
if not os.path.isdir('./results'):
    os.mkdir('./results')
savename = f"./results/{args.algorithm}-{args.problem}-{args.gamma}-{args.random_warmup}-{args.seed}-perfs.csv"

df = pd.DataFrame(tried_configs)
df["val_loss"] = performances
df.to_csv(savename)

