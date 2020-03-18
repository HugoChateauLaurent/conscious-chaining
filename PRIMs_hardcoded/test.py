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
D = 16  # the dimensionality of the vectors
D_GW = 16  # the dimensionality of the vectors in GW
GW_THR = .2
AM_THR = .2
AM_function = lambda x: x#x>0
AM_cls = spa.ThresholdingAssocMem # either WTAAssocMem or ThresholdingAssocMem
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
n_different_digits = 4 # default: 4
n_different_operations = 3 # default: 3

number_of_total_trials = n_blocks_per_operation * n_trials_per_digit * n_different_digits * n_different_operations
number_of_non_learning_trials = number_of_total_trials
number_of_learning_trials = max(0,number_of_total_trials - number_of_non_learning_trials)
print("number_of_learning_trials",number_of_learning_trials) 
print("number_of_non_learning_trials",number_of_non_learning_trials) 
print("number_of_total_trials",number_of_total_trials)


add_ON = '+ON'
keys = ['TWO','FOUR','SIX','EIGHT','X', \
               'MORE','LESS', \
    'G', 'V', 'COM', 'ADD', 'SUB', \
    'SIMPLE', 'CHAINED_ADD', 'CHAINED_SUB', \
               'ON'
    ] + ['V_COM', 'COM_PM', 'V_ADD', 'V_SUB', 'ADD_COM', 'SUB_COM']

vocab = spa.Vocabulary(dimensions=D, pointer_gen=np.random.RandomState(seed))
vocab_GW = spa.Vocabulary(dimensions=D_GW, pointer_gen=np.random.RandomState(seed))

for voc in [vocab, vocab_GW]:
    voc.populate(";".join(keys))
vocab_GW.populate(";".join([p+"_ON=ON*"+p for p in ['V', 'COM', 'ADD', 'SUB']])) # this is done to avoid similarity with other SPs

trials = xps.createTrials(n_blocks_per_operation, n_trials_per_digit, n_different_digits, n_different_operations, shuffle=True)
xp = xps.Xp1(number_of_learning_trials, trials, fixation="0")
#xp = xps.TestMasking(.183, number_of_learning_trials, trials, fixation="0")

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
    
    V = spa.State(vocab, label='V')

    # An associative memory for the "compare to 5" operation
    COM_input = spa.State(vocab, feedback=.85, feedback_synapse=.05, label='COM_input')
    COM = AM_cls(threshold=AM_THR, 
        input_vocab=vocab, mapping=
        {
            'TWO':'LESS'+add_ON,
            'FOUR':'LESS'+add_ON,
            'SIX':'MORE'+add_ON,
            'EIGHT':'MORE'+add_ON,
        },
        function=AM_function,
        label='COM',
        n_neurons = npd_AM
    )
    COM_input >> COM.input

    # A slot that combines selected information from the processors
    GW = spa.State(vocab_GW, label='GW', feedback=.6)
    
    processors = [V, COM]
    competition_keys = {
        V: ['TWO','FOUR','SIX','EIGHT','X'],
        COM: ['MORE','LESS'],
    }
    preconscious = {}
    filters = {}
    for processor in processors:
        
        preconscious[processor] = processor
        
        filters[processor] = spa.modules.WTAAssocMem(
            GW_THR,
            vocab,
            mapping={k:k+add_ON for k in competition_keys[processor]},
            function=lambda x: x>0,
            n_neurons = npd_AM,
            label="filter "+processor.label
        )
        preconscious[processor] >> filters[processor].input
        
        nengo.Connection(filters[processor].output, GW.input, 
            transform=
                     np.dot(
                         vocab_GW.parse(processor.label).get_binding_matrix(), 
                         vocab.transform_to(vocab_GW)
                     ))
    
    conscious_conditions = { # bottom-up strength * top-down attention
        V:   ROUTING_BIAS+spa.dot(preconscious[V]  .output, s.ON),
        COM: ROUTING_BIAS+spa.dot(preconscious[COM].output, s.ON)
    }
    access_labels = []
    with spa.Network(label='conscious access') :
        with spa.ActionSelection() as access:
            for proc in processors:
                access_labels.append(proc.label)
                spa.ifmax(proc.label, conscious_conditions[proc],
                            preconscious[proc] >> filters[proc],
                         )
            access_labels.append("Thresholder")
            spa.ifmax(ROUTING_BIAS+ROUTING_THR)
    
    if model_source == "GW":
        sources = {proc:spa.State(vocab, label='broadcast source '+proc.label) for proc in processors}
        for proc in sources.keys():
            nengo.Connection(GW.output, sources[proc].input, transform=
                             np.dot(
                                 vocab_GW.transform_to(vocab, populate=False),
                                 vocab_GW.parse("~"+proc.label).get_binding_matrix()
                             ))
    elif model_source == "processors":
        sources = {proc:proc if isinstance(proc,spa.State) else proc.output for proc in processors}
    action_labels = []
    with spa.Network(label='broadcast') :
        with spa.ActionSelection() as broadcast:

            action_labels.append("V_COM")
            spa.ifmax("V_COM", ROUTING_BIAS+spa.dot(GW, s.V*s.ON) * spa.dot(GW, s.G*s.SIMPLE),
                        sources[V] >> COM_input,
                     )
            
            action_labels.append("V_SUB")
            
            action_labels.append("Thresholder")
            spa.ifmax("Thresholder", ROUTING_BIAS+ROUTING_THR) # Threshold for action
    
    
    