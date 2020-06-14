import nengo
import nengo_spa as spa
import numpy as np

import sys
sys.path.append('..')
from modules import SP_to_scalar, LeakyIntegrator

s = spa.sym
dimensions = 64
vocab = spa.Vocabulary(dimensions)
vocab.populate('TWO.unitary() ; ADD1.unitary() ; ' \
    'FOUR=TWO*ADD1*ADD1 ; SIX=FOUR*ADD1*ADD1 ; EIGHT=SIX*ADD1*ADD1 ;' \
    'FIVE=FOUR*ADD1 ;'\
    'MORE ; LESS')
    
threshold = .5
decision_SPs = ['LESS','MORE']

def decision_function(x):
    if x<-threshold:
        return vocab.parse(decision_SPs[0]).v
    elif x>threshold:
        return vocab.parse(decision_SPs[1]).v
    else:
        return vocab.parse("0").v

with spa.Network() as model:
    
    A_in = spa.Transcode(input_vocab=vocab, output_vocab=vocab)
    A = SP_to_scalar(vocab)
    A_in >> A.input
    
    B_in = spa.Transcode(input_vocab=vocab, output_vocab=vocab, label="REFERENCE")
    B = SP_to_scalar(vocab)
    B_in >> B.input
    s.FIVE >> B_in
    
    evidence = LeakyIntegrator(.2,0.8,100,1)

    # compute x-reference
    nengo.Connection(A.output,evidence.input)
    nengo.Connection(B.output, evidence.input, transform=-1)
    
    output = spa.Transcode(input_vocab=vocab, output_vocab=vocab)
    nengo.Connection(evidence.output, output.input, function=decision_function)
    