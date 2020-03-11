import nengo
import nengo_spa as spa
import numpy as np

D = 64
vocab = spa.Vocabulary(D)
vocab.populate(';'.join(['A','B']))

class Button():
    def __init__(self, SP_vectors, trial_length, dt=None, thr=.5, focus_length=.5):
        self.t_last_evt = -100
        self.SP_vectors = SP_vectors
        self.t_last_step = 0
        self.dt = dt
        self.thr = thr
        self.trial_length = trial_length
        self.focus_length = focus_length
    
    def __call__(self,t,x):
        if not self.dt or t-self.dt > self.t_last_step:
            self.t_last_step = t
            if t//self.trial_length > self.t_last_evt//self.trial_length and t > (t//self.trial_length)*self.trial_length + self.focus_length:
                for i in range(len(self.SP_vectors)):
                    similarities = np.dot(self.SP_vectors,x)
                    if np.dot(x,self.SP_vectors[i]) > self.thr:
                        self.t_last_evt = t
                        return i+1
                        
        return 0
    
with spa.Network() as model:
    s = spa.State(vocab)
    inh = spa.create_inhibit_node(s)
    button = nengo.Node(Button([vocab.parse('A').v, vocab.parse('B').v], 1), size_in=D)
    
    spa.sym.B >> s
    nengo.Connection(s.output, button)