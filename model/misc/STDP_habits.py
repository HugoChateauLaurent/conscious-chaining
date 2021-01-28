import nengo
import nengo_spa as spa
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from stdp import STDP


class Cycler():
    def __init__(
        self, 
        SPs, 
        relative_start,
        relative_stop,
        SP_duration,
        t_stop
    ):
        assert relative_start <= relative_stop
        self.SPs = SPs
        self.start = relative_start * SP_duration
        self.stop = relative_stop * SP_duration
        self.SP_duration = SP_duration
        self.t_stop = t_stop
        

    def make_step(self):
        def f(t):
            t_in_window = t % self.SP_duration
            idx = int((t % (self.SP_duration*len(self.SPs))) // self.SP_duration)
            if t_in_window>self.start and t_in_window<self.stop and t<self.t_stop:
                return self.SPs[idx]
            else:
                return "0"
        return f
    
n_SPs = 4
D = 16
subdimensions = D
n_neurons = 30*subdimensions

vocab = spa.Vocabulary(D)
SPs = ['SP'+str(i) for i in range(n_SPs)]
vocab.populate(';'.join(SPs))

print(np.dot(vocab.parse('SP0').v, vocab.parse('SP1').v))

with spa.Network() as model:
    
    n_training_cycles = 5
    SP_duration = .5
    T_learning = n_training_cycles * n_SPs * SP_duration
    intercepts = nengo.dists.Uniform(.1, .1)

    A_f = Cycler(SPs, .25, .75, SP_duration, n_SPs*999).make_step()
    A = nengo.Ensemble(
        30*D,
        D,
        intercepts=nengo.dists.Uniform(.1, .1)
    )
    A_inp = spa.Transcode(A_f, output_vocab=vocab)
    nengo.Connection(A_inp.output, A)
    A_out = spa.Transcode(input_vocab=vocab, output_vocab=vocab)
    nengo.Connection(A, A_out.input)
    
    B_f = Cycler(np.roll(SPs,-1), .5, 1, SP_duration, n_SPs*n_training_cycles).make_step()
    B = nengo.Ensemble(
        30*D,
        D,
        intercepts=intercepts
    )
    B_inp = spa.Transcode(B_f, output_vocab=vocab)
    nengo.Connection(B_inp.output, B)
    B_out = spa.Transcode(input_vocab=vocab, output_vocab=vocab)
    nengo.Connection(B, B_out.input)
        
    connection = nengo.Connection(
            A.neurons,
            B.neurons,
            transform=np.zeros((B.n_neurons, A.n_neurons)),
            learning_rule_type=nengo.Oja(
                learning_rate=1e-8,
                # beta=0
                # max_weight=.1,
                # min_weight=-.1,
                # bounds="none",
            ),
            # learning_rule_type=nengo.BCM(
                # learning_rate=5e-8,
            # ),
    )
    
    connection = nengo.Connection(
            A.neurons,
            B.neurons,
            transform=np.zeros((B.n_neurons, A.n_neurons)),
            learning_rule_type=nengo.BCM(
                learning_rate=5e-11,
                # beta=0
                # max_weight=.1,
                # min_weight=-.1,
                # bounds="none",
            ),
            # learning_rule_type=nengo.BCM(
                # learning_rate=5e-8,
            # ),
    )
        