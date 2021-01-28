import nengo
import nengo_spa as spa
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

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
D = int(32)
subdimensions = D
n_neurons = 50*subdimensions

vocab = spa.Vocabulary(D)
SPs = np.array(['SP'+str(i) for i in range(n_SPs)])
vocab.populate(';'.join(SPs))

SP_vs = np.array([vocab.parse(SP).v for SP in SPs])
intercepts = [(np.dot(SP_vs[:,i*subdimensions:(i+1)*subdimensions], SP_vs[:,i*subdimensions:(i+1)*subdimensions].T) - np.eye(n_SPs)).flatten().max() for i in range(int(D/subdimensions))]
print(f"Intercepts: {intercepts}")

with spa.Network() as model:
    
    # Visual input all along the 1s trials
    Visual_f = Cycler(SPs, 0, 1, 1, np.inf).make_step()
    Visual = spa.State(vocab, represent_cc_identity=False, subdimensions=subdimensions)
    Visual_inp = spa.Transcode(Visual_f, output_vocab=vocab)
    Visual_inp >> Visual

    # Artificial slow network that provides the answer after a 500ms delay
    Slow_f = Cycler(np.roll(SPs,-1), .5, 1, 1, np.inf).make_step()
    Slow = spa.Transcode(Slow_f, output_vocab=vocab)
    
    # Fast network that learns to predict the answer of the slow net
    Fast = spa.State(vocab, represent_cc_identity=False, subdimensions=subdimensions)
    
    # Create the encoding ensembles
    encoders = [
        nengo.Ensemble(n_neurons, subdimensions, intercepts=[intercepts[i]] * n_neurons, label='Voja encoder')
        for i in range(int(D/subdimensions))
    ]
    
    # Learn the encoders/keys
    voja = nengo.Voja(learning_rate=5e-3, post_synapse=None)
    connections_in = [
        nengo.Connection(ens, encoders[i], synapse=None, learning_rule_type=voja)
        for i,ens in enumerate(Visual.all_ensembles)
    ]
    
    # Learn the decoders/values, initialized to a null function
    connections_out = [
        nengo.Connection(
            encoders[i],
            Fast.input[i*subdimensions:(i+1)*subdimensions],
            learning_rule_type=nengo.PES(1e-3),
            function=lambda x: np.zeros(subdimensions),
            synapse=None
        )
        for i in range(int(D/subdimensions))
    ]

    # Create the error populations
    errors = [nengo.Node(size_in=subdimensions, label='error') for i in range(int(D/subdimensions))]
    error_SP_readout = spa.Transcode(input_vocab=vocab, output_vocab=vocab)
    for i in range(int(D/subdimensions)):
        nengo.Connection(errors[i], error_SP_readout.input[i*subdimensions:(i+1)*subdimensions], synapse=None)
    
    # Calculate the error and use it to drive the PES rule
    for i in range(int(D/subdimensions)):
        nengo.Connection(Slow.output[i*subdimensions:(i+1)*subdimensions], errors[i], transform=-1, synapse=None)
        nengo.Connection(Fast.output[i*subdimensions:(i+1)*subdimensions], errors[i], synapse=None)
        nengo.Connection(errors[i], connections_out[i].learning_rule)
        
    
        