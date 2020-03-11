import nengo
import nengo_spa as spa

import string

D = 80
items = string.ascii_uppercase[:6]
vocab = spa.Vocabulary(D)
vocab.populate(';'.join(list(items)+['CONV']))


with spa.Network() as model:
    
    GW = spa.State(vocab, label='GW')
    source = spa.State(vocab, label='source')
    broadcast = spa.Bind(vocab, unbind_right=True)
    
    GW >> broadcast.input_left
    source >> broadcast.input_right
    
    spa.sym.A * spa.sym.CONV >> GW
    spa.sym.CONV >> source
