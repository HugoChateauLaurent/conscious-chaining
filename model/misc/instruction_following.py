import nengo
import nengo_spa as spa
from collections import deque
import numpy as np
from nengo.utils.filter_design import cont2discrete
from nengo.networks import InputGatedMemory as WM

class TwoStepsTrigger(spa.Network):
    
    def __init__(self, vocab, gain=5, deriv_synapse=.1, theta=.05, order=30, dt=.001, **kwargs):
        
        super().__init__(**kwargs)
        
        self.vocab = vocab

        # parameters of LMU
        theta = theta # length of window
        order = order # number of Legendre polynomials
        dt = dt  # simulation timestep
        
        Q = np.arange(order, dtype=np.float64)
        R = (2 * Q + 1)[:,None] / theta
        j, i = np.meshgrid(Q, Q)
        
        A = np.where(i < j, -1, (-1.0) ** (i - j + 1)) * R
        B = (-1.0) ** Q[:, None] * R
        C = np.ones((1, order))
        D = np.zeros((1,))
        
        A, B, C, D, _ = cont2discrete((A, B, C, D), dt=dt, method="zoh")
        
        with self:
            self.input = nengo.Node(size_in=1)
            
            self.deriv = nengo.Ensemble(
                100,
                1, 
                intercepts=nengo.dists.Uniform(0.1, .9),
                encoders=[[1]]*100
            )
            
            gain = 5
            nengo.Connection(self.input, self.deriv, transform=gain)
            nengo.Connection(self.input, self.deriv, transform=-gain, synapse=deriv_synapse)
            
            self.lmu = nengo.Node(size_in=order)
            nengo.Connection(self.deriv, self.lmu, transform=B, synapse=None)
            nengo.Connection(self.lmu, self.lmu, transform=A, synapse=0)
            
            self.output = nengo.Node(size_in=2)
            nengo.Connection(self.deriv, self.output[0], synapse=None)
            nengo.Connection(self.lmu, self.output[1], transform=C, synapse=None)
        
        self.declare_input(self.input, None)
        self.declare_output(self.output, None)
            

class SP_WM(spa.Network):
    def __init__(
        self,
        n_neurons, 
        vocab,
        feedback=1.0, 
        difference_gain=1.0, 
        recurrent_synapse=0.02, 
        difference_synapse=None, 
        **kwargs
    ):
        
        super(SP_WM, self).__init__(**kwargs)
        self.vocab = vocab

        with self:
            self.wm = WM(n_neurons, vocab.dimensions, feedback, difference_gain, recurrent_synapse, difference_synapse, label='WM')
        
        self.input = self.wm.input
        self.gate = self.wm.gate
        self.output = self.wm.output
            
        self.declare_input(self.input, self.vocab)
        self.declare_input(self.gate, None)
        self.declare_output(self.output, self.vocab)

D = int(64*4)
n_digits = 3
vocab = spa.Vocabulary(D)
digits = ['D'+str(i) for i in range(n_digits)]
vocab.populate(';'.join(digits))
vocab.populate('A;B;C')

def startup_f(t):
        if t<.1:
            return "D0"
        else:
            return "0"
            
with nengo.Network() as model:
    
    # utility_in = nengo.Node([0])
    # utility = spa.Scalar()
    # nengo.Connection(utility_in, utility.input)
    
    # tst = TwoStepsTrigger(vocab)


    # POS = SP_WM(50, vocab)


    INSTRUCTIONS = spa.Transcode(input_vocab=vocab, output_vocab=vocab)
    vocab.parse('D0*A + D1*B + D2*C') >> INSTRUCTIONS
    for i in range(n_digits):
        digit = vocab.parse('D'+str(i))
        digit * digit >> INSTRUCTIONS
    PRIM = spa.Bind(vocab=vocab, unbind_right=True)
    INSTRUCTIONS >> PRIM.input_left
    POS >> PRIM.input_right
    
    # AM = spa.WTAAssocMem(
    #     0, 
    #     vocab, 
    #     mapping=dict(list({'D'+str(i):'D'+str(i+1) for i in range(n_digits-1)}.items()) + list({'D'+str(n_digits-1):'D0'}.items())),
    #     function=lambda x: x>0
    # )
    # nengo.Connection(POS.output, AM.input)
    
    # ADD = SP_WM(50, vocab)
    # nengo.Connection(AM.output, ADD.input)
    # nengo.Connection(ADD.output, POS.input)


    # cmd = spa.Transcode(input_vocab=vocab, output_vocab=vocab)
    
    # startup = spa.Transcode(startup_f, output_vocab=vocab)
    # nengo.Connection(startup.output, POS.wm.mem.input, synapse=None)
    
    # bias = nengo.Node([1])
    # nengo.Connection(bias, POS.gate)
    # nengo.Connection(bias, ADD.gate)
    # tst_ens = nengo.Ensemble(100,2)
    # nengo.Connection(tst.output, tst_ens)
    # nengo.Connection(tst_ens[0], POS.gate, transform=-1, function=lambda x: x>0)
    # nengo.Connection(tst_ens[1], ADD.gate, transform=-1, function=lambda x: x>0)

    # BG_bias = 0
    # BG_thr = .2
    
    # # with spa.ActionSelection():
        
    # #     spa.ifmax(
    # #         tst.output[0],
    # #         1 >> POS.gate,
    # #     )
        
    # #     spa.ifmax(
    # #         tst.output[1],
    # #         1 >> ADD.gate,
    # #     )
        
    # #     spa.ifmax( # Threshold for action
    # #         BG_bias + BG_thr,
    # #         1 >> POS.gate,
    # #         1 >> ADD.gate,
    # #     )
        
    # with spa.ActionSelection():
        
    #     for i in range(n_digits):
    #         spa.ifmax(
    #             BG_bias + utility * spa.dot(POS, vocab.parse('D'+str(i))),
    #             vocab.parse('D'+str(i)) >> cmd,
    #             3 >> tst.input,
    #         )
    
    #     spa.ifmax( # Threshold for action
    #         BG_bias + BG_thr
    #     )
