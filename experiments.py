import nengo
import nengo_spa as spa
import numpy as np
import random
import math
from abc import ABC, abstractmethod

def create_xp(xp, n_blocks_per_operation, n_trials_per_digit, n_different_digits, n_different_operations, stim_duration, t_start, t_answer, seed, SOA=None, shuffle=True):
    trials = createTrials(xp, n_blocks_per_operation, n_trials_per_digit, n_different_digits, n_different_operations, shuffle=shuffle, rng=np.random.RandomState(seed))
    if xp == 1:
        return Xp1(trials, stim_duration=stim_duration, t_start=t_start, t_answer=t_answer)
    elif xp == 3:
        return Xp3(trials, prime_duration=stim_duration, t_start=t_start, t_answer=t_answer, SOA=SOA)


class Xp1Trial():
    def __init__(self, operation, stimulus):
        self.operation = operation
        self.stimulus = stimulus

    @property
    def N(self):
        return {'D2':2, 'D4':4, 'D6':6, 'D8':8}[self.stimulus]

    @property
    def target(self):
        N = self.N
        if self.operation == 'CHAINED_ADD':
            N += 2
        elif self.operation == 'CHAINED_SUB':
            N -= 2
        if N > 8:
            N = 2
        elif N < 2:
            N = 8
        return N

    @property
    def addend(self):
        if self.operation == 'CHAINED_ADD':
            return 'D2'
        elif self.operation == 'CHAINED_SUB':
            return '~D2'
        else:
            return '1'
    @property
    def instructions(self):
        if self.operation in ['CHAINED_ADD', 'CHAINED_SUB']:
            return '\
                D2 * (GET_ADD   + SET_ADD) +\
                D3 * (GET_COM   + SET_COM) + \
                D4 * SET_M \
            '
        else:
            return '\
                D2 * (GET_COM   + SET_COM) + \
                D3 * SET_M \
            '

    @property    
    def expected_action(self):
        return 1 + int(self.target > 5) # 0: no action, 1: LESS, 2: MORE

    @property
    def congruent(self):
        return 1 if (self.target > 5 and self.N > 5) or (self.target < 5 and self.N < 5) else -1
    
    @property
    def stimulus_idx(self):
        return {'D2':0, 'D4':1, 'D6':2, 'D8':3}[self.stimulus]

    @property
    def operation_idx(self):
        return {'SIMPLE':0, 'CHAINED_ADD':1, 'CHAINED_SUB':2}[self.operation]

class Xp3Trial(Xp1Trial):
    def __init__(self, operation, prime, target):
        self.operation = operation
        self.prime = prime
        super().__init__(operation, target)    
    

def createTrials(xp, n_blocks_per_operation, n_trials_per_digit, n_different_digits, n_different_operations, shuffle, rng):
        trials = []
        for operation in ['SIMPLE', 'CHAINED_ADD', 'CHAINED_SUB'][:n_different_operations]:
            for i in range(n_blocks_per_operation*n_trials_per_digit):
                for stimulus in ['D2', 'D4', 'D6', 'D8'][:n_different_digits]:
                    if xp == 1:
                        trials.append(Xp1Trial(operation, stimulus))
                    elif xp == 3:
                        for prime in ['D2', 'D4', 'D6', 'D8'][:n_different_digits]:
                            trials.append(Xp3Trial(operation, prime, stimulus))
        if shuffle:
            if rng is None:
                print("Warning: setting random seed")
                rng = np.random.RandomState()
            rng.shuffle(trials)

        return trials

class AbstractXp(ABC):
    def __init__(self, trial_length, trials, mask, t_start):
        self.trial_length = trial_length
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
    def RETINA_input(self,t,x):
        pass

    def INSTRUCTIONS_input(self,t,x):
        trial, t_in_trial = self(t)
        return trial.instructions
    def ADDEND_input(self,t,x):
        trial, t_in_trial = self(t)
        return trial.addend

    @property 
    def T(self):
        return self.trial_length*len(self.trials)

class Xp1(AbstractXp): # chronometric exploration   
    def __init__(self, trials, t_start=1, stim_duration=.029, t_answer=1, rng=None):
        self.stim_duration = stim_duration
        super().__init__(t_start+stim_duration+t_answer, trials, None, t_start)

    def RETINA_input(self,t,x):
        trial, t_in_trial = self(t)
        if t_in_trial < self.t_start:
            return "FIXATE"
        elif t_in_trial < self.t_start+self.stim_duration:
            return trial.stimulus
        else:
            return "0"
        
        
class Xp3(AbstractXp): # priming
    def __init__(self, trials, t_start=1, prime_duration=.029, SOA=.3, t_answer=1, rng=None):
        self.prime_duration = prime_duration
        self.SOA = SOA
        super().__init__(t_start+prime_duration+SOA+t_answer, trials, None, t_start)

    def RETINA_input(self,t,x):
        trial, t_in_trial = self(t)
        if t_in_trial < self.t_start:
            return "FIXATE"
        elif t_in_trial < self.t_start+self.prime_duration:
            return trial.prime
        elif t_in_trial < self.t_start+self.prime_duration+self.SOA:
            return "0"
        else:
            return trial.stimulus

# class TestMasking(AbstractXp):
#     def __init__(self, SOA, trials, t_start=1):
#         self.SOA = SOA # SOA in original study is in [.016,.033,.083]
#         super().__init__(2.029, trials, None, t_start)

#     def RETINA_input(self,t,x):
#         if False: # Original study
#             trial, t_in_trial = self(t)
#             if t_in_trial < self.t_start:
#                 return "FIXATE"
#             elif self.t_start < t_in_trial < self.t_start+.016:
#                 return trial.stimulus
#             elif self.t_start+self.SOA < t_in_trial < self.t_start+self.SOA+.150: # Masks during 150 ms after SOA (=stim + fixation)
#                 return "MASK"
#             else:
#                 return "0"

#         else: # Stimulus shown during SOA duration, no blank between stimulus and mask
#             trial, t_in_trial = self(t)
#             if t_in_trial < self.t_start:
#                 return "FIXATE"
#             elif self.t_start < t_in_trial < self.t_start+self.SOA:
#                 return trial.stimulus
#             else:
#                 return "0"


    
