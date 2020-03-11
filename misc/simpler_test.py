import nengo
import nengo_spa as spa
import numpy as np
from random import shuffle
import random

symbol_keys = ['TWO','FOUR','SIX','EIGHT','X', \
               'MORE','LESS', \
    'G', 'V', 'COM', 'ADD', 'SUB', 'PREV', 'PM', \
    'SIMPLE', 'CHAINED_ADD', 'CHAINED_SUB', \
               'ON'
    ] + ['V_COM', 'COM_PM', 'V_ADD', 'V_SUB', 'ADD_COM', 'SUB_COM', 'V_PM', 'FOCUS']
add_ON = '+ON'

seed = np.random.randint(999)
print("Warning: setting random seed")
np.random.seed(seed)
random.seed(seed)
s = spa.sym
D = 64  # the dimensionality of the vectors
AM_THR = .3
ROUTING_THR = .3
GW_threshold = 0

# Number of neurons (per dimension or ensemble)
scale_npds = 1
npd_AM = int(50*scale_npds) # Default: 50
npd_state = int(50*scale_npds) # Default: 50
npd_BG = int(100*scale_npds) # Default: 100
npd_thal1 = int(50*scale_npds) # Default: 50
npd_thal2 = int(40*scale_npds) # Default: 40
n_scalar = int(50*scale_npds) # Default: 50

vocab = spa.Vocabulary(dimensions=D, name='all', pointer_gen=np.random.RandomState(seed))
vocab.populate(";".join(symbol_keys))

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
    G = spa.State(vocab, label='G')
    s.SIMPLE >> G
    
    V = spa.State(vocab, label='V')
    s.TWO >> V
    
    # The previously executed PRIM
    PREV = spa.State(vocab, feedback=.8, feedback_synapse=.05, label='PREV')
    PREV_initial_input = spa.Transcode(lambda t: "FOCUS" if t<.01 else "0", output_vocab=vocab)
    PREV_initial_input >> PREV
    
    
    # A slot for the action (MORE or LESS)
    PM = spa.State(vocab, feedback=.8, feedback_synapse=.05, label='PM')
    
    # An associative memory for the + operation
    ADD_input = spa.State(vocab, feedback=.8, feedback_synapse=.05, label='ADD_input')
    ADD = spa.WTAAssocMem(threshold=AM_THR, 
        input_vocab=vocab, mapping=
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
    SUB_input = spa.State(vocab, feedback=.8, feedback_synapse=.05, label='SUB_input')
    SUB = spa.WTAAssocMem(threshold=AM_THR, 
        input_vocab=vocab, mapping=
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
    COM_input = spa.State(vocab, feedback=.8, feedback_synapse=.05, label='COM_input')
    COM = spa.WTAAssocMem(threshold=AM_THR, 
        input_vocab=vocab, mapping=
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
    GW = spa.State(vocab, neurons_per_dimension = 150, label='GW', represent_cc_identity=False)
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
        source = processor
        if GW_threshold:
            print('WARNING: did not implement add_ON with GWT yet')
            proc_threshold = spa.modules.WTAAssocMem(
                GW_threshold,
                vocab,
                mapping=competition_keys[processor],
                function=lambda x: x>0,
                n_neurons = npd_AM
            )
            processor >> proc_threshold.input
            source = proc_threshold
            
        #source * vocab_memory.parse(processor.label) >> GW
        
        nengo.Connection(source.output, GW.input, 
            transform=vocab.parse(processor.label).get_binding_matrix())
    
    
    # Definition of the actions
    # There are rules that carry out the actions, and rules that check the
    # conditions. If a condition is satisfied, check is set to YES which
    # is a condition for the actions.
    with spa.Network(label='BG-Thalamus') :
        with spa.ActionSelection() as bg_thalamus:
            # Action rules first
            #spa.ifmax("V_COM", spa.dot(GW, s.V*s.ON) * spa.dot(G, s.SIMPLE),
            spa.ifmax("V_COM", spa.dot(PREV, s.FOCUS) * spa.dot(GW, s.V*s.ON) * spa.dot(G, s.SIMPLE),
                        V >> COM_input,
                        s.V_COM >> PREV
            )
            
            spa.ifmax("V_SUB", spa.dot(PREV, s.FOCUS) * spa.dot(GW, s.V*s.ON) * spa.dot(G, s.CHAINED_SUB),
                        V >> SUB_input,
                        s.V_SUB >> PREV
            )
            
            spa.ifmax("V_ADD", spa.dot(PREV, s.FOCUS) * spa.dot(GW, s.V*s.ON) * spa.dot(G, s.CHAINED_ADD),
                        V >> ADD_input,
                        s.V_ADD >> PREV
            )
            
            spa.ifmax("COM_PM", .33*(spa.dot(PREV, s.ADD_COM)+spa.dot(PREV, s.SUB_COM)+spa.dot(PREV, s.V_COM)) * spa.dot(GW, s.COM*s.ON),
            #spa.ifmax("COM_PM", spa.dot(GW, s.COM*s.ON),
                        COM.output >> PM,
                        s.COM_PM >> PREV
            )
            
            spa.ifmax("ADD_COM", spa.dot(PREV, s.V_ADD) * spa.dot(GW, s.ADD*s.ON) * spa.dot(G, s.ADD*s.CHAINED_ADD),
            #spa.ifmax("ADD_COM", spa.dot(GW, s.ADD*s.ON),
                        ADD.output >> COM_input,
                        s.ADD_COM >> PREV
            )
            
            spa.ifmax("SUB_COM", spa.dot(PREV, s.V_SUB) * spa.dot(GW, s.SUB*s.ON) * spa.dot(G, s.SUB*s.CHAINED_SUB),
            #spa.ifmax("SUB_COM", spa.dot(GW, s.SUB*s.ON),
                        SUB.output >> COM_input,
                        s.SUB_COM >> PREV
            )
                      
                      
            spa.ifmax("FOCUS", spa.dot(PREV, s.COM_PM) * spa.dot(GW, s.PM*s.ON),
            #spa.ifmax("FOCUS", spa.dot(GW, s.PM*s.ON),
                        s.FOCUS >> PREV
            )
                      
            
            spa.ifmax("Thresholder", ROUTING_THR, s.FOCUS >> PREV) # Threshold for action
    
    
    