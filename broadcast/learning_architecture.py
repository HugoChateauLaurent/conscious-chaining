import nengo
import nengo_spa as spa
import numpy as np
from random import shuffle
import random
from weight_save import WeightSaver

    
import sys, os
import math
import matplotlib.pyplot as plt

from IPython import display


class Trial():
    def __init__(self, operation, stimulus):
        self.operation = operation
        self.stimulus = stimulus

class Experiment():
    def __init__(self, trial_length, number_of_learning_trials, trials):
        self.trial_length = trial_length
        self.number_of_learning_trials = number_of_learning_trials
        self.trials = trials

    def __call__(self, t):
        t = round(t,4) - .001 # Avoid float problems
        trial_number = math.floor(t / self.trial_length)
        t_in_trial = t - trial_number * self.trial_length
        trial = self.trials[trial_number]
        return trial, t_in_trial

    
    def RETINA_f(self, t):
        trial, t_in_trial = self(t)
        
        if 1 < t_in_trial:# < 1.029:
            return trial.stimulus
        else:
            return "0"

    def G_f(self, t):
        trial = self(t)[0]
        return trial.operation

    def CORRECT_PRIM_f(self, t):
        trial, t_in_trial = self(t)

        # before stimulus appears
        if t_in_trial < 1:
            return '0_0' # focus

        else: # ['SIMPLE', 'CHAINED_ADD', 'CHAINED_SUB']
            if trial.operation == 'SIMPLE':
                if   1 < t_in_trial < 1.33:
                    return 'V_COM'
                elif 1.33 < t_in_trial < 1.66:
                    return 'COM_PM'
                else:
                    return '0_0'

            elif trial.operation == 'CHAINED_ADD':
                if   1 < t_in_trial < 1.33:
                    return 'V_ADD'
                elif 1.33 < t_in_trial < 1.66:
                    return 'ADD_COM'
                elif 1.66 < t_in_trial < 1.99:
                    return 'COM_PM'
                else:
                    return '0_0'

            elif trial.operation == 'CHAINED_SUB':
                if   1 < t_in_trial < 1.33:
                    return 'V_SUB'
                elif 1.33 < t_in_trial < 1.66:
                    return 'SUB_COM'
                elif 1.66 < t_in_trial < 1.99:
                    return 'COM_PM'
                return '0_0'

            else:
                print("unknown operation")

    def learning_inhibit_f(self, t):
        trial_number = math.floor(t / self.trial_length)
        if trial_number < self.number_of_learning_trials:
            return 0
        else:
            return -1
        
class RandomExperiment(Experiment):
    def __init__(self, trial_length, n_blocks_per_operation=10, n_trials_per_digit=5, n_different_digits=3, n_different_operations=4):
        trials = []
        for operation in ['SIMPLE', 'CHAINED_ADD', 'CHAINED_SUB'][:n_different_operations]:
            for i in range(n_blocks_per_operation*n_trials_per_digit):
                for stimulus in ['TWO', 'FOUR', 'SIX', 'EIGHT'][:n_different_digits]:
                    trials.append(Trial(operation, stimulus))
        shuffle(trials)

        super().__init__(trial_length, number_of_learning_trials, trials)

class Button():
    def __init__(self, SP_vectors, trial_length, dt=None, thr=.5, focus_length=1):
        self.t_last_evt = -100
        self.SP_vectors = SP_vectors
        self.t_last_step = 0
        self.dt = dt
        self.thr = thr
        self.trial_length = trial_length
        self.focus_length = focus_length
    
    def __call__(self,t,x):
        if not self.dt or t-self.dt > self.t_last_step:
            self.t_last_step = t
            if t//self.trial_length > self.t_last_evt//self.trial_length and t > (t//self.trial_length)*self.trial_length + self.focus_length:
                for i in range(len(self.SP_vectors)):
                    similarities = np.dot(self.SP_vectors,x)
                    if np.dot(x,self.SP_vectors[i]) > self.thr:
                        self.t_last_evt = t
                        return i+1
                        
        return 0
        
seed = 1
np.random.seed(seed)
random.seed(seed)
s = spa.sym
D = 32  # the dimensionality of the vectors
AM_THR = .3
GW_threshold = 0

# Number of neurons (per dimension or ensemble)
scale_npds = 1
npd_AM = int(50*scale_npds) # Default: 50
npd_state = int(50*scale_npds) # Default: 50
npd_BG = int(100*scale_npds) # Default: 100
npd_thal1 = int(50*scale_npds) # Default: 50
npd_thal2 = int(40*scale_npds) # Default: 40
npd_bind = int(200*scale_npds) # Default: 200
n_scalar = int(50*scale_npds) # Default: 50

save_weights = False
load_weights = False

n_blocks_per_operation = 11 # default: 10
n_trials_per_digit = 5 # default: 5
n_different_digits = 4 # default: 4
n_different_operations = 3 # default: 3

number_of_total_trials = n_blocks_per_operation * n_trials_per_digit * n_different_digits * n_different_operations
number_of_non_learning_trials = number_of_total_trials-600
number_of_learning_trials = max(0,number_of_total_trials - number_of_non_learning_trials)
print("number_of_learning_trials",number_of_learning_trials) 
print("number_of_non_learning_trials",number_of_non_learning_trials) 
print("number_of_total_trials",number_of_total_trials)


trial_length = 2.029

T = number_of_total_trials * trial_length - .00001# simulations run a bit too long
print('T',T)

symbols = ['TWO','FOUR','SIX','EIGHT','X', \
               'MORE','LESS', \
    'G', 'V', 'COM', 'ADD', 'SUB', 'PREV', 'PM', \
    'SIMPLE', 'CHAINED_ADD', 'CHAINED_SUB'
    ]

vocab = spa.Vocabulary(dimensions=D, name='all', pointer_gen=np.random.RandomState(seed))
vocab.populate(";".join(symbols))

xp = RandomExperiment(trial_length, n_blocks_per_operation, n_trials_per_digit, n_different_digits, n_different_operations)