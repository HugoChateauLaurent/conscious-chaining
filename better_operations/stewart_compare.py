import numpy as np
import nengo
import nengo_spa as spa

from scipy import spatial

D = 128
vocab = spa.Vocabulary(D)
s = spa.sym

digits = {i+1:'D'+str(i+1) for i in range(10)}
for SP in digits.values():
    vocab.populate(SP)
    
vocab.populate('MORE ; LESS')

n_samples = 1000
train_vectors = np.zeros((n_samples, D*2))
train_coords = np.zeros((n_samples, 1))

neurons_per_dim = 50



for i in range(n_samples):
    x = np.random.randint(10) + 1
    y = np.random.randint(10) + 1
    x_SP = vocab.parse("D"+str(x)).v
    y_SP = vocab.parse("D"+str(y)).v

    train_vectors[i, :D] = vocab.parse("D"+str(x)).v
    train_vectors[i, D:] = vocab.parse("D"+str(y)).v
    train_coords[i, 0] = -1 if x<y else 1 if x>y else 0
    
threshold = .25
decision_SPs = ['LESS','MORE']

with spa.Network() as model:

    a = spa.Transcode(lambda t,x: "D5" if t<1 else "0", input_vocab=vocab, output_vocab=vocab)
    b = spa.Transcode(lambda t,x: "D2" if t<1 else "0", input_vocab=vocab, output_vocab=vocab)
    
    # s.D1 >> a
    # s.D2 >> b
    
    clean_a = spa.ThresholdingAssocMem(0,vocab,
        mapping = digits.values()
    )
    clean_b = spa.ThresholdingAssocMem(0,vocab,
        mapping = digits.values()
    )
    
    a >> clean_a
    b >> clean_b
    
    compare = nengo.Ensemble(2*D*neurons_per_dim, 2*D)
    nengo.Connection(clean_a.output, compare[:D])
    nengo.Connection(clean_b.output, compare[D:])
    
    control_signal = nengo.Ensemble(neurons_per_dim*2, 2, radius=1.5)
    for i, clean_input in enumerate([clean_a, clean_b]):
        for j in range(clean_input.selection.thresholding.output.size_out):
            nengo.Connection(clean_input.selection.thresholding.output[j], control_signal[i])
    
    controlled = nengo.Ensemble(neurons_per_dim*2, 2, radius=1.5)
    nengo.Connection(
        compare,
        controlled[0],
        function=train_coords,
        eval_points=train_vectors,
        scale_eval_points=False,
      )
    nengo.Connection(control_signal, controlled[1], function=lambda x: x[0]*x[1])
    
    
    # Create accumulator
    tau = .05
    s_evidence = 5.9
    accumulator = nengo.Ensemble(neurons_per_dim*2, 2, radius=1)
    nengo.Connection(accumulator, accumulator[0], # recurrent accumulating connection
        function=lambda x: x[0]*x[1], # conrtolled by control signal
        synapse=tau
    )      
    nengo.Connection( # controlled input
        controlled,
        accumulator[0],
        function=lambda x: x[0]*x[1],
        transform=tau*s_evidence,
        synapse=tau
      )
    nengo.Connection(control_signal, accumulator[1], function=lambda x: x[0]*x[1]) # control signal
    
    answer = spa.ThresholdingAssocMem(threshold, 
        vocab, 
        mapping=decision_SPs,
        function=lambda x: x>0
    )
    
    # answer = spa.Transcode(input_vocab=vocab, output_vocab=vocab)
    nengo.Connection(accumulator[0], answer.input, 
        function=lambda x: -x*vocab.parse(decision_SPs[0]).v+x*vocab.parse(decision_SPs[1]).v
    )
    
    
