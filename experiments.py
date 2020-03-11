import nengo
import nengo_spa as spa
import numpy as np
import random
import math
from abc import ABC, abstractmethod


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

class Trial():
    def __init__(self, operation, stimulus):
        self.operation = operation
        self.stimulus = stimulus

def createTrials(n_blocks_per_operation, n_trials_per_digit, n_different_digits, n_different_operations, shuffle):
        trials = []
        for operation in ['SIMPLE', 'CHAINED_ADD', 'CHAINED_SUB'][:n_different_operations]:
            for i in range(n_blocks_per_operation*n_trials_per_digit):
                for stimulus in ['TWO', 'FOUR', 'SIX', 'EIGHT'][:n_different_digits]:
                    trials.append(Trial(operation, stimulus))
        if shuffle:
            random.shuffle(trials)

        return trials

class AbstractXp(ABC):
    def __init__(self, trial_length, number_of_learning_trials, trials, fixation, mask):
        self.trial_length = trial_length
        self.number_of_learning_trials = number_of_learning_trials
        self.trials = trials
        self.fixation = fixation
        self.mask = mask

    def __call__(self, t):
        t = round(t,4) - .001 # Avoid float problems
        trial_number = math.floor(t / self.trial_length)
        t_in_trial = t - trial_number * self.trial_length
        trial = self.trials[trial_number]
        return trial, t_in_trial

    @abstractmethod
    def RETINA_input(self, t):
        pass

    @abstractmethod
    def G_input(self, t):
        pass
        
        
class Xp1(AbstractXp): # chronometric exploration   
    def __init__(self, number_of_learning_trials=0, trials=None, fixation="FIXATION"):
        if trials is None:
            trials = createTrials(10,5,4,3,True)
        super().__init__(2.029, number_of_learning_trials, trials, fixation, None)

    def RETINA_input(self, t):
        trial, t_in_trial = self(t)
        if t_in_trial < 1:
            return self.fixation
        elif 1 < t_in_trial < 1.029:
            return trial.stimulus
        else:
            return "0"

    def G_input(self, t):
        trial, t_in_trial = self(t)
        return trial.operation
    
class Xp2(AbstractXp): # cued-response
    pass    
class Xp3(AbstractXp): # priming
    pass    
class Xp4(AbstractXp): # masking
    pass
    

    
