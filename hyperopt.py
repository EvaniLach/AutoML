#########################################################################################
# The assignment file. While it is allowed to change this entire file, we highly
# recommend using the provided template. YOU MUST USE THE RANGES AND HYPERPARAMETERS SPECIFIED
# IN GET_RANGES AND GET_CONFIG_PERFORMAMCE (IN SHORT: USE OUR SURROGATE PROBLEMS)
#########################################################################################

import numpy as np
import argparse
import os
import pandas as pd
import math
from scipy.stats import norm,truncnorm
from scipy import stats
from collections import Counter
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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

    for j in range(function_evaluations):
        config={}
        
        for i in RANGES:
            # If there is a condition, check if it holds.
            # If not, skip this hyperparameter
            if(condition(RANGES, config, i)):
                config[i] = math.nan
                continue
            
            a = RANGES[i]['range'][0] 
            b = RANGES[i]['range'][1] 
            
            # If uniform
            if(RANGES[i]['sample']==0):
                if (RANGES[i]["type"] == 0):
                    value=np.random.choice(RANGES[i]['range']) 
                    config[i]=value  
                elif (RANGES[i]["type"] == 1):
                    value=np.random.uniform(a, b) 
                    config[i]=value
                else:
                    value=np.random.randint(a, b+1) 
                    config[i]=value
            # Else log        
            else:
                value=np.random.uniform(np.log(a), np.log(b)) 
                value = np.exp(value)
                if RANGES[i]['type'] == 2:
                    value = round(value)
                config[i]=value # exponentiate value back 

        configs.append(config)
        history.append(GET_CONFIG_PERFORMANCE(config, problem))
    #print(configs)
    return history, configs

# Function for checking if hyperparameter has a condition and whether it holds for random search
def condition(ranges, config, i):
        if('condition' in ranges[i]):
            if (ranges[i]['condition'](config) == False):
                return True

# Check if current hyperparameter is active, specifically for all 'nodes_in_layers'
def condition_check(x_star, i): 
    if 'nlayers' in x_star.keys():
        layers = x_star['nlayers']
        if i[-1].isdigit():
            if int(i[-1]) > layers:
                return True
                
# Get EI for a single value of a hyperparameter
def get_EI(samples, x, y, a, b, max_sd, max_sd_y, gamma):
    EI = []
    for j in range(len(samples)):   
        pd_lx = 0
        pd_gx = 1
        if len(x) > 0:
            pd_lx = get_pdf(samples[j], x, a, b, max_sd)     
        if len(y) > 0:
            pd_gx = get_pdf(samples[j], y, a, b, max_sd_y)
        if pd_gx == 0:
            pd_gx = 1
        # Calculate EI
        value = (gamma + (1-gamma)*(pd_lx/pd_gx))
        EI.append(value)
    return EI

def return_node_list(x):
    nodes= []
    for i in x.keys():
        if 'nodes_in_layer' in i:
            nodes.append(x[i])
    return nodes

# Returns a DF of the configurations
def to_df(configs, loss):

    hyper_parameters=pd.DataFrame(columns=["configs","loss"])
    hyper_parameters["configs"]=configs
    hyper_parameters["loss"]=loss
    
    hyper_parameters_1 = (hyper_parameters["configs"].apply(pd.Series))
    hyper_parameters_1['loss'] = hyper_parameters["loss"]
    
    return hyper_parameters_1
            
# Function for dividing samples into good and bad dataframes    
def good_bad(gamma, hyper_parameters):

    sorted_df=(hyper_parameters.sort_values(by=["loss"])).reset_index(drop=True)
    index_value=int(gamma*(sorted_df.shape[0]))
    
    good_df = sorted_df.iloc[:index_value]
    bad_df = sorted_df.iloc[index_value:sorted_df.shape[0]]
            
    return good_df, bad_df

# Sample from a truncated Gaussian
# To sample from a mixture of truncated Gaussians, we first randomly pick a Gaussian from the list of values, then we sample from this Gaussian
def sample_truncnorm(a, b, x, sd, n):
    # First pick a random Gaussian
    index = np.random.choice(range(len(x)))
    # Scipy doc mentions to transform the truncations as follows:
    a, b = (a - x[index]) / sd[index], (b - x[index]) / sd[index]
    samples = stats.truncnorm.rvs(a, b, loc=x[index], scale=sd[index], size=n)
    return samples

# Returns the probability density of a value x 
# Density is calculated by taking the sum of the densities of x in all Gaussians and dividing over number of Gaussians
# This takes a list of all x values and sigma's
def get_pdf(x_i, x, a, b, sd):
    n = len(x)
    total = 0
    # For each Gaussian, calculate density and sum them all
    for i in range(n):
        mean = x[i]
        sigma = sd[i]
        a, b = (a - mean) / sigma, (b - mean) / sigma
        total += stats.truncnorm.pdf(x_i, a, b, loc=mean, scale=sigma)
       # if error == 'error':
       #     print(x_i, a, b, mean, sigma)
    # Divide sum over number of Gaussians    
    return total/n

# We implemented the bandwith selection from Ozaki et al. 
# Reference: 'Multiobjective Tree-Structured Parzen Estimator for Computationally Expensive Optimization Problems', doi: 10.1145/3377930.3389817
def scales(x, a, b):
    if len(x) > 1:
        # Calculate distance to nearest neighbours
        diff = np.diff(x)
    else:
        diff = [0]
    # Constant to prevent very small sigma    
    epsilon = (b-a)/min(100,len(x)+2)
    scales = []
    for i in range(0, len(diff)):
        # Either greatest distance or epsilon
        max_ = max(diff[i-1], diff[i], epsilon)
        sigma = min(max_, b-a)
        scales.append(sigma)        
    scales.insert(0,min(max(diff[0], epsilon), b-a))
    scales.insert(-1,min(max(diff[-1], epsilon), b-a))
    
    return scales
# Sample uiformly
def sample_uniform(hyperparameter, n):
    sample = np.random.choice(hyperparameter['range'], n)    
    return sample

def normal_dist(x , mean , sd):
    prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density

# The density of categorical values is equal to its frequency in l(x) or g(x)
def categorical_pdf(x, samples):
    n = len(samples)
    c = Counter(x)
    densities = np.zeros(n)
    for i in range(n):
        densities[i] = c[samples[i]]
        
    return densities

def calculate_EI(lx, gx):
    # In case gx is very small or 0
    gx[gx < 0.0001] = 0.0001
    EI = lx/gx
    return EI

# n = number of candidate samples
def tpe(problem, function_evaluations=150, random_warmup=30, gamma=0.2, n=2,**kwargs):
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
    configs_tpe = []

    # get all information about the hyperparameters we want to tune for this problem
    # (see utils.py) for the form of this.  
    RANGES = GET_RANGES(problem) 

    # TODO: implement the rest of the function

    # Initial warm-up using random search
    loss,configs=random_search(problem,random_warmup)    
    # history.append(loss)
    # configs.append(configs)
    
    hyperparameters = to_df(configs, loss)    
        
    for k in range(function_evaluations):
        
        good_df, bad_df = good_bad(gamma, hyperparameters)
        # Initialize best found configuration
        x_star = {}
        
        # Iterate through each hyperparameter
        for i in good_df.columns.difference(['loss']):
            # Seperate into good and bad samples
            x = good_df[i].dropna().values.tolist()
            y = bad_df[i].dropna().values.tolist()
            # Add back together for sampling
            #full = x + y
            # Define truncation boundaries
            a = RANGES[i]['range'][0]
            b = RANGES[i]['range'][1]
            # Check if current hyperparamter is active    
            if(condition_check(x_star, i)):
                x_star[i] = math.nan
                continue
            
            # If categorical
            if RANGES[i]['type'] == 0:
                samples = sample_uniform(RANGES[i], n)
                # Calculate probability density of the samples
                lx = categorical_pdf(x, samples)
                gx = categorical_pdf(y, samples)
                EI = calculate_EI(lx, gx)
                EI_max = np.argmax(EI)
                value = samples[EI_max]
                x_star[i] = value
                
            else:
                # If sample type is log-unfirom
                if RANGES[i]['sample'] == 1:
                    x = np.log(x)
                    y = np.log(y)
                    #full = np.log(full)
                    a, b = np.log(a), np.log(b)
                    
                # Sort because we need the standard deviation to the furthest neighbour
                x.sort()                  
                y.sort()
                #full.sort()
                # Calculate sigma for getting the density from the Gaussians
                std_x = scales(x, a, b)
                # Sample from truncated gaussians
                if len(x)==0:
                    x = [0]
                samples = sample_truncnorm(a, b, x, std_x, n)
                # In case l(x) or g(x) does not contain samples: the sigma is 0
                max_sd, max_sd_y = 0, 0
                if len(x) > 0:
                    max_sd = scales(x, a, b)
                if len(y) > 0:
                    max_sd_y = scales(y, a, b)
                # Calculate EI
                EI = get_EI(samples, x, y, a, b, max_sd, max_sd_y, gamma)
                EI_max = np.argmax(EI)
                value = samples[EI_max]
                # If sample type is log-uniform, exponentiate
                if RANGES[i]['sample'] == 1:
                    value = np.exp(value)
                # If hp type is int, round
                if RANGES[i]['type'] == 2:
                    value = round(value)      
                x_star[i] = value
        # Calculate loss and append to observations
        configs_tpe.append(x_star)
        x_star['loss'] = GET_CONFIG_PERFORMANCE(x_star, problem) 
        history.append(GET_CONFIG_PERFORMANCE(x_star, problem))
        hyperparameters = hyperparameters.append(x_star, ignore_index=True)            
   
    #print(configs)
    return history, configs_tpe

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
