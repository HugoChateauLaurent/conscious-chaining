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
from modules import Processor, Button

import math
import matplotlib.pyplot as plt
from cycler import cycler
default_cycler = cycler('color', ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF'])
plt.rc('axes', prop_cycle=(default_cycler))

from IPython import display

def get_senders(processors):
    return [p for p in processors if p.sender]

if True: # random seed
    seed = np.random.randint(999)
    print("Warning: setting random seed")
else:
    seed = 1
    
np.random.seed(seed)
random.seed(seed)
s = spa.sym
D = 128  # the dimensionality of the vectors
D_GW = 128  # the dimensionality of the vectors in GW
PROC_FDBCK = .85
GW_FDBCK = .85
ROUTING_THR = 0
ROUTING_BIAS = 0

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
n_different_digits = 2 # default: 4
n_different_operations = 3 # default: 3

number_of_total_trials = n_blocks_per_operation * n_trials_per_digit * n_different_digits * n_different_operations
number_of_non_learning_trials = number_of_total_trials
number_of_learning_trials = max(0,number_of_total_trials - number_of_non_learning_trials)
print("number_of_learning_trials",number_of_learning_trials) 
print("number_of_non_learning_trials",number_of_non_learning_trials) 
print("number_of_total_trials",number_of_total_trials)


keys = ['TWO','FOUR','SIX','EIGHT', \
        'FIXATE', 'DISTRACT', \
               'MORE','LESS', \
    'SIMPLE', 'CHAINED_ADD', 'CHAINED_SUB']

vocab = spa.Vocabulary(dimensions=D, pointer_gen=np.random.RandomState(seed), max_similarity=.02)
vocab_GW = spa.Vocabulary(dimensions=D_GW, pointer_gen=np.random.RandomState(seed), max_similarity=.02)

for voc in [vocab, vocab_GW]:
    voc.populate(";".join(keys))
vocab_GW.populate('SOURCE ; CONTENT ; TYPE ; DIGIT ;'+
                  'G ; V ; COM ; ADD ; SUB')
vocab_GW.populate(";".join([p+"_SOURCE=(SOURCE*"+p+").normalized()" for p in ['V', 'COM', 'ADD', 'SUB']])) # this is done to avoid similarity with other SPs

trials = xps.createTrials(n_blocks_per_operation, n_trials_per_digit, n_different_digits, n_different_operations, shuffle=True)
xp = xps.Xp1(number_of_learning_trials, trials, fixation="FIXATE")
#xp = xps.TestMasking(.083, number_of_learning_trials, trials, fixation="0")

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
    
    inputs_net = spa.Network(label='inputs')

    # We start defining the buffer slots in which information can
    # be placed:
    
    # A slot for the goal/task
    G = spa.State(vocab, label='G')
    with inputs_net:
        G_input = spa.Transcode(xp.G_input, output_vocab = vocab)
    G_input >> G
    
    # A slot for the visual input (the digit N). Feedback is used for iconic memory (100-300ms)
    V_mapping = {
        digit:'CONTENT*'+digit+' + TYPE*DIGIT' for digit in ['TWO','FOUR','SIX','EIGHT']
    }
    V_mapping.update({
        control:'TYPE*'+control for control in ['FIXATE','DISTRACT']
    })
    """RETINA = spa.WTAAssocMem(
        0.1,
        vocab,
        mapping=V_mapping.keys(),
        function=lambda x: x>0,
        n_neurons = npd_AM
    )"""
    with inputs_net:
        RETINA_input = spa.Transcode(xp.RETINA_input,output_vocab = vocab)
        
    V = Processor(
        vocab, vocab_GW, 'V', V_mapping, 
        receiver=False,
        npd_AM=npd_AM, seed=seed
    )
    nengo.Connection(RETINA_input.output, V.input_state.input, synapse=None)
#     RETINA.output >> V.input_state
    
    # A slot for the action (MORE or LESS)
    PM = Processor(
        vocab, vocab, 'PM',
        AM_mapping = ['MORE','LESS'],
        AM_cls=spa.WTAAssocMem,
        sender=False, add_ON=False,
        npd_AM=npd_AM, seed=seed
    )
    """PM = spa.State(vocab, feedback=PROC_FDBCK, label='PM')
    with spa.Network(label='Action'):
        with nengo.Network() as ACT_net:
            ACT_net.config[nengo.Ensemble].neuron_type = nengo.Direct()
            ACT = spa.State(vocab, label='ACT direct')
        with spa.ActionSelection():            
                spa.ifmax( spa.dot(PM, s.MORE),
                            s.MORE >> ACT)
                spa.ifmax( spa.dot(PM, s.LESS),
                            s.LESS >> ACT)
                spa.ifmax( .3)"""
            
    BTN = nengo.Node(Button([vocab.parse('MORE').v, vocab.parse('LESS').v], xp.trial_length), size_in=D)
    nengo.Connection(PM.AM.output, BTN)

    # An associative memory for the + operation
    ADD = Processor(
        vocab, vocab_GW, 'ADD',
        AM_mapping = {
            'TWO':'FOUR*CONTENT',
            'FOUR':'SIX*CONTENT',
            'SIX':'EIGHT*CONTENT',
            'EIGHT':'TWO*CONTENT'
        },
        npd_AM=npd_AM, seed=seed
    )
    
    # An associative memory for the - operation
    SUB = Processor(
        vocab, vocab_GW, 'SUB',
        AM_mapping = {
            'TWO':'EIGHT*CONTENT',
            'FOUR':'TWO*CONTENT',
            'SIX':'FOUR*CONTENT',
            'EIGHT':'SIX*CONTENT'
        },
        npd_AM=npd_AM, seed=seed
    )
    
    # An associative memory for the "compare to 5" operation
    COM = Processor(
        vocab, vocab_GW, 'COM',
        AM_mapping = {
            'TWO':'LESS*CONTENT',
            'FOUR':'LESS*CONTENT',
            'SIX':'MORE*CONTENT',
            'EIGHT':'MORE*CONTENT'
        },
        npd_AM=npd_AM, seed=seed
    )

    # A slot that combines selected information from the processors
    GW = spa.State(vocab_GW, label='GW', feedback=GW_FDBCK)
    
    processors = [V, ADD, SUB, COM, PM]
    for p in processors:
        if p.sender:
            p.filter >> GW
    
    # access network
    access_labels = []
    with spa.Network(label='conscious access') :
        with spa.ActionSelection() as access:
            for p in processors:
                if p.sender:
                    access_labels.append(p.label)
                    spa.ifmax(p.label, ROUTING_BIAS+spa.dot(p.preconscious, s.SOURCE)*p.attention,
                                spa.translate(p.preconscious, vocab_GW) + vocab_GW.parse(p.label)*s.SOURCE >> p.filter.input,
                             )
            access_labels.append("Thresholder")
            spa.ifmax(ROUTING_BIAS+ROUTING_THR)   
    
    # broadcast network
    with spa.Network(label='broadcast'):
        for p in processors: # each processor p receives GW's content if p's "receive" level is more than a threshold
            if p.receiver:
                with spa.ActionSelection() as broadcast:
                    spa.ifmax("GO", p.receive,
                                 spa.translate(GW * ~s.CONTENT, vocab) >> p.input_state
                             )
                    spa.ifmax("NOGO", .5)
       
    # routing network
    routing_labels = []
    with spa.Network(label='routing'):
        with spa.ActionSelection() as routing:
            
            routing_labels.append("ATTEND")
            spa.ifmax("V_COM", ROUTING_BIAS+spa.dot(GW, s.FIXATE*s.TYPE),
                          *(+1/2 >> p.attention if p==V else -1/2 >> p.attention for p in get_senders(processors))
                     )
            
            routing_labels.append("V_COM")
            spa.ifmax("V_COM", ROUTING_BIAS+spa.dot(GW, s.DIGIT*s.TYPE) * spa.dot(G, s.SIMPLE),
                          1 >> COM.receive,
                          *(+1/2 >> p.attention if p==COM else -1/2 >> p.attention for p in get_senders(processors))
                     )

            routing_labels.append("V_SUB")
            spa.ifmax("V_SUB", ROUTING_BIAS+spa.dot(GW, s.DIGIT*s.TYPE) * spa.dot(G, s.CHAINED_SUB),
                          1 >> SUB.receive,
                          *(+1/2 >> p.attention if p==SUB else -1/2 >> p.attention for p in get_senders(processors))
                     )

            routing_labels.append("V_ADD")
            spa.ifmax("V_ADD", ROUTING_BIAS+spa.dot(GW, s.DIGIT*s.TYPE) * spa.dot(G, s.CHAINED_ADD),
                          1 >> ADD.receive,
                          *(+1/2 >> p.attention if p==ADD else -1/2 >> p.attention for p in get_senders(processors))
                     )

            routing_labels.append("ADD_COM")
            spa.ifmax("ADD_COM", ROUTING_BIAS+spa.dot(GW, s.ADD*s.SOURCE),
                          1 >> COM.receive,
                          *(+1/2 >> p.attention if p==COM else -1/2 >> p.attention for p in get_senders(processors))
                     )

            routing_labels.append("SUB_COM")
            spa.ifmax("SUB_COM", ROUTING_BIAS+spa.dot(GW, s.SUB*s.SOURCE),
                          1 >> COM.receive,
                          *(+1/2 >> p.attention if p==COM else -1/2 >> p.attention for p in get_senders(processors))
                     )

            routing_labels.append("COM_PM")
            spa.ifmax("COM_PM", ROUTING_BIAS+spa.dot(GW, s.COM*s.SOURCE),
                          1 >> PM.receive,
                          *(-1/2 >> p.attention for p in get_senders(processors))
                     )            

            routing_labels.append("Thresholder")
            spa.ifmax("Thresholder", ROUTING_BIAS+ROUTING_THR) # Threshold for action


