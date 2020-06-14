import nengo_spa as spa
import nengo
import numpy as np

vocab = spa.Vocabulary(64*3)
vocab.populate("ADD_1.unitary()")
s = spa.sym

# def exponent_f(reference):
#     def f(s):
#         e = np.fft.ifft(np.fft.fft(s.v) ** e).real
#         print(e)
#         return e
#     return f

def exponent_f(s, reference):
    e = np.log(np.fft.fft(s.v)) / np.log(np.fft(reference.v))
    print(e)
    return e
reference = vocab.parse("ADD_1")
e = exponent_f(reference*reference, reference)

# def f(s):
#     reference = vocab.parse("ADD_1")
#     e = np.fft.fft(s.v) ** e).real
#     print(e)
#     return e

with spa.Network() as model:
    
    trans = spa.Transcode(input_vocab=vocab, output_vocab=vocab)
    state = spa.State(vocab)
    trans >> state
    s.ADD_1 * s.ADD_1 >> trans
    
    exponent = nengo.Node(size_in=1)
    # nengo.Connection(trans.output, exponent, function=exponent_f(vocab.parse("ADD_1")))