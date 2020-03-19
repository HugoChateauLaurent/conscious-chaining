import nengo
import nengo_spa as spa
import numpy as np

vocab = spa.Vocabulary(128, max_similarity=.0001)
vocab.populate('ADD2 ; ZERO')
vocab.populate(
    'TWO = (ZERO*ADD2).normalized() ;'+\
    'FOUR = (TWO*ADD2).normalized() ;'+\
    'SIX = (FOUR*ADD2).normalized() ;'+\
    'EIGHT = (SIX*ADD2).normalized()'
)

with spa.Network() as model:
    
    
    x = spa.State(vocab, label='x')
    y = spa.State(vocab, label='y')
    spa.sym.SIX >> x
    spa.sym.TWO >> y
    
    
    # x+y
    addition = spa.Bind(vocab, label='addition')
    x >> addition.input_left
    y * ~spa.sym.ZERO >> addition.input_right
    
    # x-y
    substraction = spa.Bind(vocab, unbind_right=True, label='substraction')
    x >> substraction.input_left
    y * ~spa.sym.ZERO >> substraction.input_right
    
    
    
        
    