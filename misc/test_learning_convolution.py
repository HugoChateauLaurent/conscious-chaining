import nengo
import nengo_spa as spa

import string

D = 16
T_learning = 3
items = string.ascii_uppercase[:6]
vocab = spa.Vocabulary(D)
vocab.populate(';'.join(list(items)+['CONV']))

def cycle_array(x1, x2, T_learning, period, add_to_output="", dt=0.001):
    """Cycles through the elements"""
    i_every = int(round(period / dt))
    if i_every != period / dt:
        raise ValueError("dt (%s) does not divide period (%s)" % (dt, period))

    def f(t):
        x = x1
        if T_learning and t>T_learning:
            x = x2
        i = int(round((t - dt) / dt))  # t starts at dt
        return x[int(i / i_every) % len(x)]+add_to_output

    return f

with spa.Network() as model:
    
    inp = spa.Transcode(cycle_array(items[:-2], items[:-2], T_learning, 0.25), output_vocab=vocab)
    x = nengo.Ensemble(50*D,D)
    nengo.Connection(inp.output, x)

    y = nengo.Ensemble(50*D,D)
    conn = nengo.Connection(x, y, function=lambda x: [0]*D, learning_rule_type=nengo.PES(learning_rate=1e-3))
    error = nengo.Node(lambda t,x: x if t<T_learning else 0*x, size_in=D, size_out=D)
    nengo.Connection(error, conn.learning_rule, synapse=None)
    nengo.Connection(y, error, synapse=None)


    y_decode = spa.Transcode(input_vocab=vocab, output_vocab=vocab)
    nengo.Connection(y, y_decode.input)
    
    

    truth = spa.Transcode(cycle_array(items[:-2], items[:-2], T_learning, 0.25), output_vocab=vocab)
    nengo.Connection(truth.output, error, transform=-1, synapse=None)
    
    