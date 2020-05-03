import nengo
import nengo_spa as spa
import numpy as np

with spa.Network() as model:
    
    
    x =             nengo.Ensemble(50, 1)
    reference =     nengo.Ensemble(50, 1)
    inp = nengo.Node([0,0])
    nengo.Connection(inp[0], x)
    nengo.Connection(inp[1], reference)
    
    
    
    evidence = nengo.networks.Integrator(.1,100,1)

    # compute x-reference
    nengo.Connection(x,evidence.input)
    nengo.Connection(reference, evidence.input, transform=-1)
    
    decision = spa.networks.selection.Thresholding(500,2,.5, function=lambda x:x>0)
    nengo.Connection(evidence.output, decision.input[0])
    nengo.Connection(evidence.output, decision.input[1], transform=-1)


    
        
        
        
    