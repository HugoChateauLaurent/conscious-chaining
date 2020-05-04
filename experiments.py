import nengo
import nengo_spa as spa
import numpy as np
import random
import math
from abc import ABC, abstractmethod


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
    def __init__(self, trial_length, number_of_learning_trials, trials, mask, t_start):
        self.trial_length = trial_length
        self.number_of_learning_trials = number_of_learning_trials
        self.trials = trials
        self.mask = mask
        self.t_start = t_start

    def __call__(self, t):
        t = round(t,4) - .001 # Avoid float problems
        trial_number = math.floor(t / self.trial_length)
        t_in_trial = t - trial_number * self.trial_length
        trial = self.trials[trial_number]
        return trial, t_in_trial

    @abstractmethod
    def RETINA_input(self, t):
        pass

    def G_input(self, t):
        trial, t_in_trial = self(t)
        return trial.operation


        
        
class Xp1(AbstractXp): # chronometric exploration   
    def __init__(self, number_of_learning_trials=0, trials=None, t_start=1, stim_scale=1, fix_scale=1, stim_duration=.029):
        if trials is None:
            trials = createTrials(10,5,4,3,True)
        self.stim_duration = stim_duration
        super().__init__(2+stim_duration, number_of_learning_trials, trials, None, t_start)

    def RETINA_input(self, t):
        trial, t_in_trial = self(t)
        if t_in_trial < self.t_start:
            return "FIXATE"
        elif self.t_start < t_in_trial < self.t_start+self.stim_duration:
            return trial.stimulus
        else:
            return "0"

class TestMasking(AbstractXp):
    def __init__(self, SOA, number_of_learning_trials=0, trials=None, t_start=1):
        self.SOA = SOA # SOA in original study is in [.016,.033,.083]
        if trials is None:
            trials = createTrials(10,5,4,3,True)
        super().__init__(2.029, number_of_learning_trials, trials, None, t_start)

    def RETINA_input(self, t):
        if False: # Original study
            trial, t_in_trial = self(t)
            if t_in_trial < self.t_start:
                return "FIXATE"
            elif self.t_start < t_in_trial < self.t_start+.016:
                return trial.stimulus
            elif self.t_start+self.SOA < t_in_trial < self.t_start+self.SOA+.150: # Masks during 150 ms after SOA (=stim + fixation)
                return "MASK"
            else:
                return "0"

        else: # Stimulus shown during SOA duration, no blank between stimulus and mask
            trial, t_in_trial = self(t)
            if t_in_trial < self.t_start:
                return "FIXATE"
            elif self.t_start < t_in_trial < self.t_start+self.SOA:
                return trial.stimulus
            else:
                return "0"

    
    
class Xp2(AbstractXp): # cued-response
    pass    
class Xp3(AbstractXp): # priming
    pass    
class Xp4(AbstractXp): # masking
    pass
    

    
