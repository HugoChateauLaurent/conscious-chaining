import nengo
import nengo_spa as spa
import numpy as np

vocab_a = spa.Vocabulary(64)
vocab_b = spa.Vocabulary(64)
for v in [vocab_a, vocab_b]:
    v.populate('A;B;C')
vocab_a.populate('D')
with spa.Network() as model:
    a = spa.State(vocab_a)
    b = spa.State(vocab_b)
    nengo.Connection(a.output, b.input, transform=np.dot(vocab_b.parse('A').get_binding_matrix(),vocab_a.transform_to(vocab_b, populate=False)))
