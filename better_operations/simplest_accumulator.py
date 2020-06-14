import numpy as np
import nengo
import nengo_spa as spa

from scipy import spatial

D = 32
vocab = spa.Vocabulary(D)

digits = {i+1:'D'+str(i+1) for i in range(10)}
for SP in digits.values():
    vocab.populate(SP)
    
vocab.populate('MORE ; LESS')
decision_SPs = ['MORE','LESS']

neurons_per_dim = 200
    
def decision_function(x):
    threshold = .7
    if x<-threshold:
        return vocab.parse(decision_SPs[0]).v
    elif x>threshold:
        return vocab.parse(decision_SPs[1]).v
    else:
        return vocab.parse("0").v    

with spa.Network() as model:
    
    
    if False:
        inp = nengo.Node([0])
        control = nengo.Node([0])
        
        accumulator = nengo.Ensemble(10*neurons_per_dim*2, 2, radius=1.5)
        
        nengo.Connection(inp, accumulator[0])
        nengo.Connection(control, accumulator[1])
        
        answer = spa.Transcode(input_vocab=vocab, output_vocab=vocab)
        nengo.Connection(accumulator[0], answer.input, function=decision_function, synapse=.05, transform=.5)
    
    
    else:
        inp = nengo.Node([0])
        control = nengo.Node([0])
        
        tau = .4

        input_ens = nengo.Ensemble(neurons_per_dim*2, 2, radius=1.5)
        nengo.Connection(inp, input_ens[0])
        nengo.Connection(control, input_ens[1])
        
        accumulator = nengo.Ensemble(neurons_per_dim*2, 2, radius=1.5)
        nengo.Connection(input_ens, accumulator[0], function=lambda x: x[0]*x[1], 
            synapse=tau,
            transform=tau
        )
        nengo.Connection(control, accumulator[1])
        
        nengo.Connection(accumulator, accumulator[0], 
            function=lambda x: x[0]*x[1],
            synapse=tau, 
        )

        answer = spa.Transcode(input_vocab=vocab, output_vocab=vocab)
        nengo.Connection(accumulator[0], answer.input, function=decision_function, synapse=.05, transform=.5)
    
    
