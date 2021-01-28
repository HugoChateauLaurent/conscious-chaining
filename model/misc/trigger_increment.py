import nengo
import nengo_spa as spa
from collections import deque
import numpy as np
from nengo.utils.filter_design import cont2discrete

import sys
sys.path.append("..")
from modules import TwoStepsTrigger, WM
from vocabs import create_vocabs

D = 128
n_digits = 10
vocab = create_vocabs(D)['digits']

def POSITION_in_f(t):
        if t<.1:
            return "D0"
        else:
            return "0"


with spa.Network() as model:
    
    utility = nengo.Node([0])
    
    trigger = TwoStepsTrigger(50)

    
    output_cmd = spa.Transcode(input_vocab=vocab, output_vocab=vocab)
    
    with spa.ActionSelection():
        spa.ifmax(
            utility,
            1 >> trigger.input
        )
        
        spa.ifmax(
            .5,
        )

    POSITION = WM(100, vocab, label='position')
    INCREMENT = WM(100, vocab, label='added')
    nengo.Connection(POSITION.output, INCREMENT.input, transform=vocab.parse('D1').get_binding_matrix())
    nengo.Connection(INCREMENT.output, POSITION.input)

    with spa.ActionSelection():
        
        spa.ifmax(trigger.output[0], 
            1 >> POSITION.gate,
        )
        
        spa.ifmax(trigger.output[1],
            1 >> INCREMENT.gate
        )
        
        spa.ifmax(.5,
            1 >> POSITION.gate,
            1 >> INCREMENT.gate
        )
        
        # startup
        spa.ifmax(
            nengo.Node(lambda t: 2 if t<.1 else 0),
            spa.sym.D1 >> POSITION.input,
            1 >> INCREMENT.gate
        )