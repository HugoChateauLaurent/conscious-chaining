import nengo_spa as spa
import nengo

def xp(t):
    t_in_trial = t%2.029
    if t_in_trial < 1:
        return ".25*FIX"
    elif 1 < t_in_trial < 1.029:
        return "0.5*STIM"
    else:
        return "0"
        
stim_scale = 5
stim_duration = .029
def f(t):
    stim_scale = 1
    t_trial = t#%2+stim_duration
    if t_trial<1:
        return str(stim_scale)+"*A"
    elif t_trial<1+stim_duration:
        return str(stim_scale)+"*B"
    else:
        return "0"
    
dim = 128

with spa.Network() as model:
    
    """t = spa.Transcode(xp, output_vocab=64)
    s = spa.State(64, feedback=.8, feedback_synapse=.005)
    t>>s"""


    """s2 = spa.State(64, feedback=.8, feedback_synapse=.005)
    
    scalar = spa.Scalar()
    
    with spa.ActionSelection() as as_net:
        spa.ifmax(0.5)
        spa.ifmax(spa.dot(s2,spa.sym.A)>0.5, 2>>scalar)
        spa.ifmax(spa.dot(s2,spa.sym.B), -2>>scalar)
        spa.ifmax(0.5)"""
        
        
    # s = spa.State(dim, feedback=0, feedback_synapse=.05)
    
    add_ON = "+V"
    
    s = spa.ThresholdingAssocMem(
        threshold=0, 
        input_vocab=dim, 
        output_vocab=dim,
        mapping={k:k+add_ON for k in ['A','B']},
        function=lambda x:x,
    )
    t = spa.Transcode(f, output_vocab=dim)
    t>>s.input
    
    
    
    GW_content = spa.WTAAssocMem(
        threshold=0, 
        input_vocab=dim, 
        output_vocab=dim,
        mapping=['A','B'],
        function=lambda x:x>0,
    )
    s * stim_scale >> GW_content.input
    
    
    for ens in GW_content.all_ensembles:
        print(ens)
        nengo.Connection(ens, ens, transform=.9, synapse=.05)
    
    #nengo.Connection(GW_content.output, GW_content.input, transform=.7, synapse=.1)
    #GW_content.output * 0.5 >> GW_content.input
    
    
    """n = nengo.Node([0])
    s = spa.Scalar()
    nengo.Connection(n, s.input)
    nengo.Connection(s.output, s.input, transform=1)"""
    