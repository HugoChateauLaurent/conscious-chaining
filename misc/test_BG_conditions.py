import nengo
import nengo_spa as spa

vocab = spa.Vocabulary(32)
vocab.populate('A;B;C')

with spa.Network() as model:
    A = spa.State(vocab, label="A")    
    B = spa.State(vocab, label="B")
    C = spa.State(vocab, label="C")
    
    with spa.ActionSelection() as bg_thalamus:
        spa.ifmax((spa.dot(A,spa.sym.A)+spa.dot(B, spa.sym.B)) * spa.dot(C, spa.sym.C))