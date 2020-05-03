import nengo_spa as spa
import nengo
import numpy as np


vocab = 16

def norm_f(x):
    print(x)
    print(np.linalg.norm(x))
    return np.linalg.norm(x)

with spa.Network() as model:
    
    s = spa.State(vocab)
    
    norm = nengo.Node(size_in=1)
    s.output.output = lambda t,x: x
    nengo.Connection(s.output, norm, function=norm_f)
    
    
    sp = nengo.Node(size_in=vocab)
    nengo.Connection(s.output, sp)
    norm_sp = nengo.Node(size_in=1)
    sp.output = lambda t,x: x
    nengo.Connection(sp, norm_sp, function=norm_f)
    
    
    
    
    
    
    zero = spa.semantic_pointer.Zero(vocab).v
    print(np.linalg.norm(zero))