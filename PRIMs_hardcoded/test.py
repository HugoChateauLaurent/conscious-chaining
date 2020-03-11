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
import math
import matplotlib.pyplot as plt

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

    
    def RETINA_input(self, t):
        trial, t_in_trial = self(t)
        
        if 1 < t_in_trial:# < 1.029:
            return trial.stimulus
        else:
            return "0"

    def G_input(self, t):
        trial = self(t)[0]
        return trial.operation
        
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
        
        
seed = np.random.randint(999)
print("Warning: setting random seed")
np.random.seed(seed)
random.seed(seed)
s = spa.sym
D = 64  # the dimensionality of the vectors
AM_THR = .3
ROUTING_THR = .3
GW_threshold = 0
PRIM_feedback = 1

# Number of neurons (per dimension or ensemble)
scale_npds = .75
npd_AM = int(50*scale_npds) # Default: 50
npd_state = int(50*scale_npds) # Default: 50
npd_BG = int(100*scale_npds) # Default: 100
npd_thal1 = int(50*scale_npds) # Default: 50
npd_thal2 = int(40*scale_npds) # Default: 40
n_scalar = int(50*scale_npds) # Default: 50

n_blocks_per_operation = 1 # default: 10
n_trials_per_digit = 1 # default: 5
n_different_digits = 3 # default: 4
n_different_operations = 1 # default: 3

number_of_total_trials = n_blocks_per_operation * n_trials_per_digit * n_different_digits * n_different_operations
number_of_non_learning_trials = number_of_total_trials-600
number_of_learning_trials = max(0,number_of_total_trials - number_of_non_learning_trials)
print("number_of_learning_trials",number_of_learning_trials) 
print("number_of_non_learning_trials",number_of_non_learning_trials) 
print("number_of_total_trials",number_of_total_trials)


trial_length = 2.029

T = number_of_total_trials * trial_length - .00001# simulations run a bit too long
print('T',T)

symbol_keys = ['TWO','FOUR','SIX','EIGHT','X', \
               'MORE','LESS', \
    'G', 'V', 'COM', 'ADD', 'SUB', 'PREV', 'PM', \
    'SIMPLE', 'CHAINED_ADD', 'CHAINED_SUB', \
               'ON'
    ]
add_ON = '+ON'

prim_keys = ['V_COM', 'COM_PM', 'V_ADD', 'V_SUB', 'ADD_COM', 'SUB_COM', 'V_PM', 'FOCUS']
all_keys = symbol_keys + prim_keys
vocab_memory = spa.Vocabulary(dimensions=D, name='all', pointer_gen=np.random.RandomState(seed))
vocab_memory.populate(";".join(all_keys))
prim_vocab = vocab_memory.create_subset(prim_keys)


xp = RandomExperiment(trial_length, n_blocks_per_operation, n_trials_per_digit, n_different_digits, n_different_operations)


np.random.seed(seed)
random.seed(seed)

model = spa.Network(seed=seed)
with model:
    
    model.config[spa.State].neurons_per_dimension = npd_state
    #model.config[spa.WTAAssocMem].n_neurons = npd_AM # Doesn't work -> set for individual AM
    model.config[spa.Scalar].n_neurons = n_scalar
    model.config[spa.BasalGanglia].n_neurons_per_ensemble = npd_BG
    model.config[spa.Thalamus].neurons_action = npd_thal1
    model.config[spa.Thalamus].neurons_channel_dim = npd_thal1
    model.config[spa.Thalamus].neurons_gate = npd_thal2

    # We start defining the buffer slots in which information can
    # be placed:
    
    # A slot for the goal/task
    G = spa.State(vocab_memory, label='G')
    
    # A slot for the visual input (the digit N). Feedback is used for iconic memory (100-300ms)
    RETINA = spa.WTAAssocMem(
        0.1,
        vocab_memory,
        mapping={k:k+add_ON for k in ['TWO','FOUR','SIX','EIGHT','X']},
        function=lambda x: x>0,
        n_neurons = npd_AM
    )
    nengo.Connection(RETINA.input, RETINA.input, transform=.85, synapse=.005)
    V = spa.State(vocab_memory, label='V')
    nengo.Connection(RETINA.output, V.input, synapse=.055)
    
    # The previously executed PRIM
    PREV = spa.State(vocab_memory, feedback=.8, feedback_synapse=.05, label='PREV')
    
    # A slot for the action (MORE or LESS)
    PM = spa.State(vocab_memory, feedback=.8, feedback_synapse=.05, label='PM')
    with nengo.Network() as ACT_net:
        ACT_net.config[nengo.Ensemble].neuron_type = nengo.Direct()
        ACT = spa.State(vocab_memory, label='ACT direct')

    # An associative memory for the + operation
    ADD_input = spa.State(vocab_memory, feedback=.8, feedback_synapse=.05, label='ADD_input')
    ADD = spa.WTAAssocMem(threshold=AM_THR, 
        input_vocab=vocab_memory, mapping=
        {
            'TWO':'FOUR'+add_ON,
            'FOUR':'SIX'+add_ON,
            'SIX':'EIGHT'+add_ON,
            'EIGHT':'TWO'+add_ON,
        },
        function=lambda x: x>0,
        label='ADD',
        n_neurons = npd_AM
    )
    ADD_input >> ADD.input
    
    # An associative memory for the - operation
    SUB_input = spa.State(vocab_memory, feedback=.8, feedback_synapse=.05, label='SUB_input')
    SUB = spa.WTAAssocMem(threshold=AM_THR, 
        input_vocab=vocab_memory, mapping=
        {
            'TWO':'EIGHT'+add_ON,
            'FOUR':'TWO'+add_ON,
            'SIX':'FOUR'+add_ON,
            'EIGHT':'SIX'+add_ON,
        },
        function=lambda x: x>0,
        label='SUB',
        n_neurons = npd_AM
    )
    SUB_input >> SUB.input
    
    # An associative memory for the "compare to 5" operation
    COM_input = spa.State(vocab_memory, feedback=.8, feedback_synapse=.05, label='COM_input')
    COM = spa.WTAAssocMem(threshold=AM_THR, 
        input_vocab=vocab_memory, mapping=
        {
            'TWO':'LESS'+add_ON,
            'FOUR':'LESS'+add_ON,
            'SIX':'MORE'+add_ON,
            'EIGHT':'MORE'+add_ON,
        },
        function=lambda x: x>0,
        label='COM',
        n_neurons = npd_AM
    )
    COM_input >> COM.input

    # A slot that combines selected information from the processors
    GW = spa.State(vocab_memory, neurons_per_dimension = 150, label='GW')
    processors = [G, V, PREV, PM, ADD, SUB, COM]
    competition_keys = {
        G: ['SIMPLE', 'CHAINED_ADD', 'CHAINED_SUB'],
        V: ['TWO','FOUR','SIX','EIGHT','X'],
        PREV: ['V_COM', 'COM_PM', 'V_ADD', 'V_SUB', 'ADD_COM', 'SUB_COM', 'V_PM', 'FOCUS'],
        PM: ['MORE','LESS'],
        ADD: ['TWO','FOUR','SIX','EIGHT'],
        SUB: ['TWO','FOUR','SIX','EIGHT'],
        COM: ['MORE','LESS'],
    }
    for processor in processors:
        source = processor.output
        if GW_threshold:
            print('WARNING: did not implement add_ON with GWT yet')
            proc_threshold = spa.modules.WTAAssocMem(
                GW_threshold,
                vocab_memory,
                mapping=competition_keys[processor],
                function=lambda x: x>0,
                n_neurons = npd_AM
            )
            processor >> proc_threshold.input
            source = proc_threshold.output
            
        #source * vocab_memory.parse(processor.label) >> GW
        
        nengo.Connection(source, GW.input, 
            transform=vocab_memory.parse(processor.label).get_binding_matrix())
    
    # Create the inputs
    with spa.Network(label='inputs'):
        #RETINA_input = spa.Transcode(xp.RETINA_input,output_vocab = vocab_memory)
        RETINA_input = spa.Transcode(input_vocab=vocab_memory,output_vocab = vocab_memory)
        #G_input = spa.Transcode(xp.G_input, output_vocab = vocab_memory)
        G_input = spa.Transcode(input_vocab=vocab_memory,output_vocab = vocab_memory)
        s.SIMPLE >> G_input
        s.TWO >> RETINA_input
        
        

    nengo.Connection(RETINA_input.output, RETINA.input, synapse=None)
    G_input >> G
    
    # Definition of the actions
    # There are rules that carry out the actions, and rules that check the
    # conditions. If a condition is satisfied, check is set to YES which
    # is a condition for the actions.
    with spa.Network(label='BG-Thalamus') :
        with spa.ActionSelection() as bg_thalamus:
            # Action rules first
            spa.ifmax("V_COM", (spa.dot(PREV, s.FOCUS) + PRIM_feedback*spa.dot(PREV, s.V_COM) - spa.dot(V,COM_input)) * spa.dot(GW, s.V*s.ON) * spa.dot(G, s.SIMPLE),
                        V >> COM_input,
                        s.V_COM >> PREV
                     )
            
            spa.ifmax("V_SUB", (spa.dot(PREV, s.FOCUS) + PRIM_feedback*spa.dot(PREV, s.V_SUB) - spa.dot(V,SUB_input)) * spa.dot(GW, s.V*s.ON) * spa.dot(G, s.CHAINED_SUB),
                        V >> SUB_input,
                        s.V_SUB >> PREV
                     )
            
            spa.ifmax("V_ADD", (spa.dot(PREV, s.FOCUS) + PRIM_feedback*spa.dot(PREV, s.V_ADD) - spa.dot(V,ADD_input)) * spa.dot(GW, s.V*s.ON) * spa.dot(G, s.CHAINED_ADD),
                        V >> ADD_input,
                        s.V_ADD >> PREV
                     )
            
            spa.ifmax("COM_PM", (spa.dot(PREV, s.ADD_COM)+spa.dot(PREV, s.SUB_COM)+spa.dot(PREV, s.V_COM)+PRIM_feedback*spa.dot(PREV, s.COM_PM) - spa.dot(COM.output,PM)) * spa.dot(GW, s.COM*s.ON),
                        COM.output >> PM,
                        s.COM_PM >> PREV
                     )
            
            spa.ifmax("ADD_COM", (spa.dot(PREV, s.V_ADD)+PRIM_feedback*spa.dot(PREV, s.ADD_COM) - spa.dot(ADD.output,COM_input)) * spa.dot(GW, s.ADD*s.ON) * spa.dot(G, s.ADD*s.CHAINED_ADD),
                        ADD.output >> COM_input,
                        s.ADD_COM >> PREV
                     )
                      
            spa.ifmax("SUB_COM", (spa.dot(PREV, s.V_SUB)+PRIM_feedback*spa.dot(PREV, s.SUB_COM) - spa.dot(SUB.output, COM_input)) * spa.dot(GW, s.SUB*s.ON) * spa.dot(G, s.SUB*s.CHAINED_SUB),
                        SUB.output >> COM_input,
                        s.SUB_COM >> PREV
                     )
                      
                      
            """spa.ifmax("FOCUS", (spa.dot(PREV, s.COM_PM)+PRIM_feedback*spa.dot(PREV, s.FOCUS)) * spa.dot(GW, s.PM*s.ON),
                        s.FOCUS >> PREV
                     )"""
                      
            
            spa.ifmax("Thresholder", ROUTING_THR, s.FOCUS >> PREV) # Threshold for action
    
    
    with spa.Network(label='Action'):
        with spa.ActionSelection():            
                spa.ifmax( spa.dot(PM, s.MORE),
                            s.MORE >> ACT)
                spa.ifmax( spa.dot(PM, s.LESS),
                            s.LESS >> ACT)

                spa.ifmax( AM_THR)
            
    BTN = nengo.Node(Button([vocab_memory.parse('MORE').v, vocab_memory.parse('LESS').v], trial_length), size_in=D)
    nengo.Connection(ACT.output, BTN)
    