import nengo
import nengo_spa as spa
import numpy as np
from random import shuffle
# import nengo_ocl
import sys, os
import math

from IPython import display
from nengo_gui.ipython import IPythonViz

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

    def V_input(self, t):
        trial, t_in_trial = self(t)
        SOA = .08
        if t_in_trial < .016:
            return trial.stimulus
        elif SOA < t_in_trial < SOA + .150: # Masks during 150 ms after SOA (=stim + fixation)
            #return "0"
            return "X"#-0.5*"+trial.stimulus
        else:
            return "0"
        
class RandomExperiment(Experiment):
    def __init__(self, trial_length, n_blocks_per_operation, n_digits_per_block):
        trials = []
        for operation in ['SIMPLE', 'CHAINED_ADD', 'CHAINED_SUB']:
            for i in range(n_blocks_per_operation*n_digits_per_block):
                for stimulus in ['TWO', 'FOUR', 'SIX', 'EIGHT']:
                    trials.append(Trial(operation, stimulus))
        shuffle(trials)
        
        super().__init__(trial_length, number_of_learning_trials, trials)


        
s = spa.sym
D = 64  # the dimensionality of the vectors
AM_THR = .3
ROUTING_THR = 0#.3
GW_threshold = 0.5

n_blocks_per_operation = 5
n_digits_per_block = 4
number_of_total_trials = n_blocks_per_operation * n_digits_per_block * 4 * 3
number_of_non_learning_trials = 40
number_of_learning_trials = number_of_total_trials - number_of_non_learning_trials
print(number_of_learning_trials, number_of_non_learning_trials, number_of_total_trials)

trial_length = .5

T = number_of_total_trials * trial_length - .00001# simulations run a bit too long
print(T)

symbol_keys = {'TWO','FOUR','SIX','EIGHT','X','V'}
vocab_memory = spa.Vocabulary(dimensions=D, name='all')
vocab_memory.populate(";".join(symbol_keys))

    
model = spa.Network(seed=12) # was 14
with model:
    

    # We start defining the buffer slots in which information can
    # be placed:
    
    # A slot for the visual input (the digit N). Feedback is used for iconic memory (100-300ms)
    #retina = spa.State(vocab_memory, feedback=.85, feedback_synapse=.005, label='retina')
    retina = spa.modules.WTAAssocMem(
        0.1,
        vocab_memory,
        mapping=['TWO','FOUR','SIX','EIGHT','X'],
        function=lambda x: x>0,
        n_neurons = 50
    )
    nengo.Connection(retina.input, retina.input, transform=.85, synapse=.005)
    V = spa.State(vocab_memory, label='V')
    nengo.Connection(retina.output, V.input, synapse=.055)
    
    retina_in = spa.Transcode(input_vocab=vocab_memory, output_vocab=vocab_memory)
    nengo.Connection(retina.input, retina_in.input, synapse=None) 
    
    
    # A slot that combines selected information from the processors
    #GW = spa.State(vocab_memory, neurons_per_dimension = 150, label='GW')
    processors = [V]
    competition_keys = {V: ['TWO','FOUR','SIX','EIGHT','X']}
    for processor in processors:
        source = processor.output
        if GW_threshold:
            proc_threshold = spa.modules.WTAAssocMem(
                GW_threshold,
                vocab_memory,
                mapping=competition_keys[processor],
                function=lambda x: x>0
            )
            processor >> proc_threshold.input
            source = proc_threshold.output
            
    #    nengo.Connection(source, GW.input, 
    #        transform=vocab_memory.parse(processor.label).get_binding_matrix())
    
    xp = RandomExperiment(trial_length, number_of_learning_trials, number_of_total_trials)
    
    # Create the inputs
    with spa.Network(label='inputs'):
        V_input = spa.Transcode(xp.V_input,output_vocab = vocab_memory)

    nengo.Connection(V_input.output, retina.input, synapse=None)
