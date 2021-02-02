import nengo
import nengo_spa as spa
import numpy as np
import sys

# Import our classes
sys.path.append('..')
from modules import *
from vocabs import *

BG_bias = .5
BG_thr = .5
D = 64
vocab = create_vocab(D, key_range(1,11,1) + ['D0=D1*~D1'] + ['GET_V ; GET_COM ; GET_ADD ; SET_COM ; SET_ADD ; SET_M'])
                
with spa.Network() as model:
    
    input_INSTRUCTIONS = spa.Transcode(
        lambda t: '\
                D1 * (GET_V) +\
                D2 * (GET_ADD + SET_ADD) +\
                D3 * (GET_COM + SET_COM) + \
                D4 * (SET_M) \
            ',
        output_vocab=vocab
    )
    POS = WM(100, vocab)
    clean_POS = spa.WTAAssocMem(
        threshold=.2,
        input_vocab=POS.vocab,
        mapping=['D1','D2','D3','D4'],
        n_neurons=50,
        function=lambda x: x>0
    )
    nengo.Connection(POS.output, clean_POS.input)
    INCREMENT = WM(100, vocab)

    PRIM = spa.Bind(neurons_per_dimension=200, vocab=vocab, unbind_right=True)
    GET_PRIM = spa.WTAAssocMem(
        threshold=.5,
        input_vocab=PRIM.vocab,
        mapping=['GET_V', 'GET_COM', 'GET_ADD'],
        n_neurons=50,
        function=lambda x: x>0
    )
    SET_PRIM = spa.WTAAssocMem(
        threshold=.5,
        input_vocab=PRIM.vocab,
        mapping=['SET_COM', 'SET_ADD', 'SET_M'],
        n_neurons=50,
        function=lambda x: x>0
    )
    PRIM >> GET_PRIM
    PRIM >> SET_PRIM

    input_INSTRUCTIONS >> PRIM.input_left
    spa.translate(clean_POS, vocab) >> PRIM.input_right
    
    SET_exec = spa.Transcode(input_vocab=vocab, output_vocab=vocab)
    GET_exec = spa.Transcode(input_vocab=vocab, output_vocab=vocab)
    
    # GET selector
    with spa.Network(label='GET selector') as GET_selector:
        GET_selector.labels = []
        with spa.ActionSelection() as GET_selector.AS:
            
            GET_selector.labels.append("GET V (FIXATE)")
            spa.ifmax(GET_selector.labels[-1], BG_bias + FIXATE_detector,
                V.preconscious >> GW.AMs[V].input,
                s.D1 >> POS.input,
                s.D1 * clean_POS >> INCREMENT.input
            )
            
            # GET_selector.labels.append("GET V")
            # spa.ifmax(GET_selector.labels[-1], BG_bias + spa.dot(GET_PRIM, s.GET_V) * (1-spa.dot(GET_exec, s.GET_V)),
            #     s.GET_V >> GET_exec,
            #     1 >> POS.gate,
            # )
    
            GET_selector.labels.append("GET ADD")
            spa.ifmax(GET_selector.labels[-1], BG_bias + spa.dot(GET_PRIM, s.GET_ADD) * (1-GW.detectors[ADD]),
                ADD.preconscious >> GW.AMs[ADD].input,
                1 >> POS.gate,
                s.D1 * clean_POS >> INCREMENT.input
            )
            
            GET_selector.labels.append("GET COM")
            spa.ifmax(GET_selector.labels[-1], BG_bias + spa.dot(GET_PRIM, s.GET_COM) * (1-GW.detectors[COM]),
                COM.preconscious >> GW.AMs[COM].input,
                1 >> POS.gate,
                s.D1 * clean_POS >> INCREMENT.input
            )
    
            GET_selector.labels.append("Thresholder")
            spa.ifmax(GET_selector.labels[-1], BG_bias + BG_thr,
                1 >> INCREMENT.gate,
                INCREMENT >> POS.input
    
            )
    
    # SET selector
    with spa.Network(label='SET selector', seed=seed) as SET_selector:
        SET_selector.labels = []
        with spa.ActionSelection() as SET_selector.AS:
    
            # SET_selector.labels.append("SET ADD")
            # spa.ifmax(SET_selector.labels[-1], BG_bias + spa.dot(SET_PRIM, s.SET_ADD) * (1-GW.detectors[ADD]),
            #     GW.AMs[COM] >> ADD.broadcast,
            #     GW.AMs[V] >> ADD.broadcast,
            # )
            
            SET_selector.labels.append("SET COM")
            spa.ifmax(SET_selector.labels[-1], BG_bias + spa.dot(SET_PRIM, s.SET_COM) * (1-GW.detectors[COM]),
                # GW.AMs[ADD] >> COM.broadcast,
                GW.AMs[V] >> COM.broadcast,
            )
    
            SET_selector.labels.append("SET M")
            spa.ifmax(SET_selector.labels[-1], BG_bias + spa.dot(SET_PRIM, s.SET_M),
                GW.AMs[COM] >> M.broadcast,
                GW.AMs[V] >> M.broadcast,
                # GW.AMs[ADD] >> M.broadcast,
            )
    
            SET_selector.labels.append("Thresholder")
            spa.ifmax(SET_selector.labels[-1], BG_bias + BG_thr) # Threshold for action
    