import nengo 
import nengo_spa as spa

vocab = spa.Vocabulary(64)
vocab.populate('A;B;C;YES')
with spa.Network() as model:
    inp = spa.State(vocab)
    AM = spa.WTAAssocMem(.5, mapping=vocab.keys(), input_vocab=vocab)
    inp >> AM.input
    
    out = spa.State(vocab)
    
    scalar = spa.Scalar()
    1/2>>scalar
    
    scalar2 = spa.Scalar()
    1/2>>scalar2
    with spa.ActionSelection() as broadcast:

        spa.ifmax(spa.dot(AM, spa.sym.A),
                    spa.sym.YES >> out,
                    *(+1/2 >> s if s==scalar else -1/2 >> s for s in [scalar,scalar2])
                 )
        spa.ifmax(.5)
    