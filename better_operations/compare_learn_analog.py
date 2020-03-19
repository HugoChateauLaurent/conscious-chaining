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
    
    
    x = spa.State(vocab, label='x')
    


    if False:
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
        
    it = Iterator()
    x_in = spa.Transcode(it.x, output_vocab=vocab, label='input')
    x_in >> x
    
    y = nengo.Node(it.y, label='expected')
    error = nengo.Ensemble(50,1)
    y_hat = nengo.Ensemble(100,1)
    nengo.Connection(y,error, transform=-1)
    nengo.Connection(y_hat,error)
    inhib_learning = nengo.Node(lambda t: 2 if t > 20 else 0)
    nengo.Connection(inhib_learning, error.neurons, transform=[[-1]] * error.n_neurons)
    
    for ens in x.all_ensembles:
        conn = nengo.Connection(ens, y_hat, function=lambda x:0)
        conn.learning_rule_type = nengo.PES()
        nengo.Connection(error, conn.learning_rule)
        
        
        
    