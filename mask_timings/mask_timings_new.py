import nengo
import nengo_spa as spa
import numpy as np
from random import shuffle
import random


use_ocl = True
if use_ocl:
    import nengo_ocl
    simulator = nengo_ocl.Simulator
else:
    simulator = nengo.Simulator
    
import sys, os

sys.path.append('..')
import experiments as xps

import math
import matplotlib.pyplot as plt
from cycler import cycler
default_cycler = cycler('color', ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF'])
plt.rc('axes', prop_cycle=(default_cycler))

from IPython import display

if True: # random seed
    seed = np.random.randint(999)
    print("Warning: setting random seed")
else:
    seed = 1
    
np.random.seed(seed)
random.seed(seed)
s = spa.sym
D = 128  # the dimensionality of the vectors
D_GW = 128*6  # the dimensionality of the vectors in GW
AM_THR = .2
AM_function = lambda x: x#x>0
AM_cls = spa.ThresholdingAssocMem # spa.WTAAssocMem
ROUTING_THR = .25
ROUTING_BIAS = .5
model_source = ["processors","GW"][1]

# Number of neurons (per dimension or ensemble)
scale_npds = 1
npd_AM = int(50*scale_npds) # Default: 50
npd_state = int(50*scale_npds) # Default: 50
npd_BG = int(100*scale_npds) # Default: 100
npd_thal1 = int(50*scale_npds) # Default: 50
npd_thal2 = int(40*scale_npds) # Default: 40
n_scalar = int(50*scale_npds) # Default: 50

n_blocks_per_operation = 1 # default: 10
n_trials_per_digit = 1 # default: 5
n_different_digits = 1 # default: 4
n_different_operations = 2 # default: 3

number_of_total_trials = n_blocks_per_operation * n_trials_per_digit * n_different_digits * n_different_operations
number_of_non_learning_trials = number_of_total_trials
number_of_learning_trials = max(0,number_of_total_trials - number_of_non_learning_trials)
print("number_of_learning_trials",number_of_learning_trials) 
print("number_of_non_learning_trials",number_of_non_learning_trials) 
print("number_of_total_trials",number_of_total_trials)


add_ON = '+ON'
keys = ['TWO','FOUR','SIX','EIGHT','X',
               'ON']

vocab = spa.Vocabulary(dimensions=D, pointer_gen=np.random.RandomState(seed))
vocab_GW = spa.Vocabulary(dimensions=D_GW, pointer_gen=np.random.RandomState(seed))

for voc in [vocab, vocab_GW]:
    voc.populate(";".join(keys))

trials = xps.createTrials(n_blocks_per_operation, n_trials_per_digit, n_different_digits, n_different_operations, shuffle=True)
#xp = xps.Xp1(number_of_learning_trials, trials, fixation="0")
xp = xps.TestMasking(.019, number_of_learning_trials, trials, fixation="0", start=0)

T = number_of_total_trials * xp.trial_length - .00001# simulations run a bit too long
print('T',T)

np.random.seed(seed)
random.seed(seed)

model = spa.Network(seed=seed)
with model:
    
    model.config[spa.State].neurons_per_dimension = npd_state
    model.config[spa.Scalar].n_neurons = n_scalar
    model.config[spa.BasalGanglia].n_neurons_per_ensemble = npd_BG
    model.config[spa.Thalamus].neurons_action = npd_thal1
    model.config[spa.Thalamus].neurons_channel_dim = npd_thal1
    model.config[spa.Thalamus].neurons_gate = npd_thal2

    # We start defining the buffer slots in which information can
    # be placed:
    
    
    # A slot for the visual input (the digit N). Feedback is used for iconic memory (100-300ms)
    RETINA = spa.WTAAssocMem(
        0.1,
        vocab,
        mapping={k:k+add_ON for k in ['TWO','FOUR','SIX','EIGHT','X']},
        #mapping=['TWO','FOUR','SIX','EIGHT','X'],
        function=lambda x: x>0,
        n_neurons = npd_AM
    )
    nengo.Connection(RETINA.input, RETINA.input, transform=.85, synapse=.005)
    V = spa.State(vocab, label='V')
    nengo.Connection(RETINA.output, V.input, synapse=.055)  
    
    processors = [V]
    competition_keys = {
        V: ['TWO','FOUR','SIX','EIGHT','X'],
        
    }
    preconscious = {}
    conscious = {}
    for processor in processors:
        """preconscious[processor] = spa.modules.WTAAssocMem(
            GW_threshold,
            vocab,
            mapping={k:k+add_ON for k in competition_keys[processor]},
            function=lambda x: x>0,
            n_neurons = npd_AM
        )
        processor >> preconscious[processor].input"""
        preconscious[processor] = processor
        
        conscious[processor] = spa.State(vocab)
    
    conscious_conditions = { # bottom-up strength * top-down attention
        V: ROUTING_BIAS+spa.dot(preconscious[V].output, s.ON),
    }
    with spa.Network(label='conscious access') :
        with spa.ActionSelection() as access:
            for proc in processors:
                spa.ifmax(proc.label, conscious_conditions[proc],
                            preconscious[proc] >> conscious[proc],
                         )
            spa.ifmax(.5)
    
    # Create the inputs
    with spa.Network(label='inputs'):
        RETINA_input = spa.Transcode(xp.RETINA_input,output_vocab = vocab)

    nengo.Connection(RETINA_input.output, RETINA.input, synapse=None)
