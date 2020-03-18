import nengo 
import nengo_spa as spa

vocab = spa.Vocabulary(64)
vocab.populate('A;B;C;YES')
with spa.Network() as model:
    inp = spa.State(vocab)
    AM = spa.WTAAssocMem(.5, mapping=vocab.keys(), input_vocab=vocab)
    inp >> AM.input
    
    out = spa.State(vocab)
    
    with spa.ActionSelection() as broadcast:

        spa.ifmax(spa.dot(AM, spa.sym.A),
                    spa.sym.YES >> out,
                 )
        spa.ifmax(.5)
    