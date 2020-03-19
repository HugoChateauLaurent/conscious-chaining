import nengo
import nengo_spa as spa
import numpy as np

vocab = spa.Vocabulary(128*2, max_similarity=.0001)
vocab.populate('ADD2 ; ZERO')
vocab.populate(
    'TWO = (ZERO*ADD2).normalized() ;'+\
    'FOUR = (TWO*ADD2).normalized() ;'+\
    'SIX = (FOUR*ADD2).normalized() ;'+\
    'EIGHT = (SIX*ADD2).normalized()'
)

class Iterator():
    def __init__(self, syms=['TWO','FOUR','SIX','EIGHT'], chg_every=.25, dt=0.001):
        self.syms = syms
        self.chg_every = chg_every
        self.last_chg = 0
        self.i = np.random.randint(len(syms))
    def x(self, t):
        if t-self.last_chg > self.chg_every:
            self.last_chg = t
            self.i = np.random.randint(len(self.syms))
        return self.syms[self.i]
    def y(self, t):
        return {'TWO':-1,'FOUR':-.33,'SIX':.33,'EIGHT':1}[self.syms[self.i]]
        


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
    
    decision = spa.networks.selection.Thresholding(100,2,.5, function=lambda x:x>0)
    nengo.Connection(evidence.output, decision.input[0])
    nengo.Connection(evidence.output, decision.input[1], transform=-1)


    
        
        
        
    