import nengo_spa as spa
import nengo
import numpy as np

dimensions = 32
vocab = spa.Vocabulary(dimensions)
vocab.populate("BASE.unitary()")

scale = 1.00001 # used to avoid log(1)
base = vocab.parse("BASE").v

def power(base, e): # as in: https://forum.nengo.ai/t/vector-exponentials-with-unitary-semantic-pointers/505/2
    return np.fft.ifft(np.fft.fft(scale*base) ** e)

def retrieve_e(y):
    e = np.median(np.log(np.absolute(np.fft.fft(y))) / np.log(np.absolute(np.fft.fft(scale*base))))
    if np.isinf(e): # happens while building with y=0
        return 0
    else:
        return e
        
def intermediate(y):
    intermediate = np.log(np.absolute(np.fft.fft(y)))
    if np.isinf(intermediate).any():
        return y*0
    else:
        return intermediate
    


with spa.Network() as model:
    
    e = nengo.Node([0])
    base_node = nengo.Node(base)
    exponentiated_base = nengo.Node(lambda t,x: power(x[:dimensions],x[-1]), size_in=dimensions+1)
    nengo.Connection(base_node, exponentiated_base[:dimensions])
    nengo.Connection(e, exponentiated_base[-1])
    
    exponentiated_base_ens = nengo.Ensemble(50*dimensions, dimensions)
    nengo.Connection(exponentiated_base, exponentiated_base_ens)

    intermediate_ens = nengo.Ensemble(50*dimensions, dimensions)
    nengo.Connection(exponentiated_base_ens, intermediate_ens, function=intermediate)

    intermediate_node = nengo.Node(lambda t,x: intermediate(x), size_in=dimensions)
    nengo.Connection(exponentiated_base, intermediate_node)
    
    intermediate_base_node = nengo.Node(lambda t,x: intermediate(x), size_in=dimensions)
    nengo.Connection(base_node, intermediate_base_node)

    x_ens = nengo.Ensemble(100*dimensions, dimensions)
    nengo.Connection(exponentiated_base, x_ens)
    
    # retrieved_e = nengo.Ensemble(100,1,radius=10)
    # nengo.Connection(x_ens, retrieved_e, function=retrieve_e)

    retrieved_e_node = nengo.Node(size_in=1)
    retrieved_e_node.output = lambda t,x: x
    nengo.Connection(exponentiated_base, retrieved_e_node, function=retrieve_e)
    
    
    # product = nengo.networks.Product(100, dimensions)
    product_node = nengo.Node(lambda t,x: x[:dimensions]*x[dimensions:], size_in=dimensions*2)
    nengo.Connection(intermediate_node, product_node[:dimensions])
    nengo.Connection(intermediate_base_node, product_node[dimensions:], function=lambda x: 1/x if (x!=0).any() else x*0)
        