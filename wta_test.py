import nengo
import nengo_spa as spa
import numpy as np


n_choices = 10
vocab = spa.Vocabulary(96)
vocab.populate(';'.join(['D'+str(d) for d in range(n_choices)]))

with spa.Network() as model:
    
    SP_vs = np.array([vocab.parse(k).v for k in vocab.keys()])
    rho_ext = nengo.Node([0]*n_choices)
    
    wta = spa.WTAAssocMem(0, vocab, mapping=vocab.keys(), function=lambda x: x>0)
    nengo.Connection(rho_ext, wta.input, transform=SP_vs.swapaxes(0,1))
    nengo.Connection(wta.output, wta.input,  # feedback
		transform=.5, synapse=.01)
		
    # wta = nengo.Node(lambda t,x: np.maximum(0,x), size_in=n_choices)
    # nengo.Connection(rho_ext, wta)#, transform=SP_vs.swapaxes(0,1))
    # nengo.Connection(
    #     wta,
    #     wta,
    #     transform=1 * (np.eye(n_choices) - 1.0),
    #     synapse=.005,
    # )
    # nengo.Connection(wta, wta,  # feedback
    # 	transform=.5, synapse=.01, function=lambda x: x>0)
		
#     wta = spa.networks.selection.Thresholding(1000, n_choices, threshold=0, function=lambda x: x>0)
#     # wta = nengo.Node(lambda t,x: np.maximum(0,x), size_in=n_choices, size_out=n_choices)
#     nengo.Connection(
#         wta.thresholded,
#         wta.input,
#         transform=1 * (np.eye(n_choices) - 1.0),
#         synapse=.005,
#     )
#     nengo.Connection(rho_ext, wta.input)
#     nengo.Connection(wta.output, wta.input,  # feedback
# 		transform=.5, synapse=.005)#, function=lambda x: x>0)