import nengo_spa as spa
import nengo
import numpy as np

dimensions = 32
vocab = spa.Vocabulary(dimensions)
vocab.populate("BASE.unitary()")

scale = 1.5 # used to avoid log(1)
base = vocab.parse("BASE").v

def power(base, e):
    return np.fft.ifft(np.fft.fft(scale*base) ** e)

def retrieve_e(y):
    e = np.median(np.log(np.absolute(np.fft.fft(y))) / np.log(np.absolute(np.fft.fft(scale*base))))
    if np.isinf(e): # happens while building with y=0
        return 0
    else:
        return e


with spa.Network() as model:
    
    e = nengo.Node(lambda t: np.sin(4*t)*10)
    base_node = nengo.Node(base)
    exponentiated_base = nengo.Node(lambda t,x: power(x[:dimensions],x[-1]), size_in=dimensions+1)
    nengo.Connection(base_node, exponentiated_base[:dimensions])
    nengo.Connection(e, exponentiated_base[-1])
    
    x_ens = nengo.Ensemble(100*dimensions, dimensions)
    nengo.Connection(exponentiated_base, x_ens)
    
    retrieved_e = nengo.Ensemble(100,1,radius=10)
    nengo.Connection(x_ens, retrieved_e, function=retrieve_e)

    retrieved_e_node = nengo.Node(size_in=1)
    nengo.Connection(x_ens, retrieved_e_node, function=retrieve_e)
    
    