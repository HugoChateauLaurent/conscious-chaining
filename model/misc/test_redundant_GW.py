import nengo
import nengo_spa as spa
import sys
sys.path.append('..')
from modules import GlobalWorkspace

n_processors=3
vocab = spa.Vocabulary(64)
vocab.populate('A;B;C')

with spa.Network() as model:
    
    Ps = [spa.State(vocab, label='proc '+str(i)) for i in range(n_processors)]
    P_inputs = [spa.Transcode(output_vocab=vocab, label=str(i)+' in') for i in range(n_processors)]
    for i in range(n_processors):
        P_inputs[i] >> Ps[i]
    
    
    GW = GlobalWorkspace(
        vocabs = {Ps[0]:vocab, Ps[1]:vocab, Ps[2]:vocab},
        mappings = {Ps[0]:['A','B','C'], Ps[1]:['A','B'], Ps[2]:['A']}
    )
    

    for i in range(n_processors):
        Ps[i] >> GW.WTAs[Ps[i]]

