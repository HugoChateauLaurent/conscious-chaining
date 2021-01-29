import nengo
import nengo_spa as spa
import numpy as np
from modules import *
from data import Data, SackurData
from vocabs import create_vocabs
import random
import pandas as pd
import matplotlib.pyplot as plt

import pytry

def make_n_neurons_per_dim(n_neurons_scale, n_neurons_scale_combined):
        # Number of neurons
        n_neurons_per_dim = { # initialize to default values then scale
            module: int(number*n_neurons_scale) for module,number in { 
            'AM': 50,
            'Ensemble': 50,
            'State': 50,
            'Bind': 200,
            'BG': 100,
            'thal_1': 50,
            'thal_2': 40,
            'Scalar': 50 # this is for the whole module (one dim)
        }.items()}
        n_neurons_per_dim['combined'] = int(50*n_neurons_scale_combined)

        return n_neurons_per_dim

class ExperimentRun(pytry.NengoTrial):
    def params(self):
        self.param('vocabs', vocabs=None)
        self.param('experiment', xp=None)
        self.param('number of samples for comparison function', n_samples=1000)
        self.param('whether to reset the comparison integrator at the beginning of each trial during 200ms', integrator_reset=True)
        self.param('scaling of the number of neurons', n_neurons_scale=1)
        self.param('scaling of the number of neurons in combined ensemble', n_neurons_scale_combined=1)
        self.param('processor feedback', proc_feedback=.8)
        self.param('time constant of comparison integrator (if 0, then use AMProcessor)', compare_tau=.05)
        self.param('threshold for action selection networks', BG_thr=.1)
        self.param('bias for action selection networks', BG_bias=.5)

    def model(self, p):
        n_neurons_per_dim = self.make_n_neurons_per_dim(p.n_neurons_scale, p.n_neurons_scale_combined)
        model = Model(
            p.vocabs, 
            p.xp, 
            p.n_samples, 
            p.integrator_reset,
            p.proc_feedback,
            p.compare_tau,
            p.BG_thr, 
            p.BG_bias, 
            n_neurons_per_dim, 
            p.seed, 
            p.plt
        )
        model.make_probes()
        self.probes = model.probes
        return model.network

    def evaluate_behaviour(self, sim, p, seed):

        sujs = []
        sess = []
        rules = []
        stims = []
        accs = [] # 0:wrong answer / 1:correct answer
        RTs = []
        cgs = []
        targets = []
        model_actions = []

        convert_rule = {'SIMPLE': 0, 'CHAINED_SUB': -2, 'CHAINED_ADD': 2}

        t = 0
        while t<p.xp.T-.01:
            t += p.xp.trial_length
            trial = p.xp(t)[0]

            sujs.append(seed)
            sess.append(2)
            rules.append(convert_rule[trial.operation])
            stims.append(trial.N)
            cgs.append(trial.congruent)
            targets.append(trial.target)

            expected_action = trial.expected_action
            t_window = (np.where(np.logical_and(sim.trange() < t, sim.trange() > t-p.xp.trial_length))[0],)
            
            # get model's action
            model_behaviour = sim.data[self.probes['BTN']][t_window]
            if np.count_nonzero(model_behaviour) > 1:
                raise ValueError("more than one action")
            
            model_action = model_behaviour.sum()
            model_actions.append(int(model_action))
            if model_action == 0:
                RTs.append((p.xp.trial_length - p.xp.t_start)*1000)
                accs.append(0)
            else:
                action_t_idx = np.nonzero(model_behaviour[:,0])
                RTs.append((sim.trange()[t_window][action_t_idx][0] - (t-p.xp.trial_length) - p.xp.t_start)*1000)
                accs.append(int(model_action==expected_action))
        
        data = np.stack((sujs, sess, rules, stims, accs, RTs, cgs, targets, model_actions), axis=1)
            
        return dict(data=Data(pd.DataFrame(data, columns=['suj','ses','rule','stim','acc','rt','cg','target','action'])))

    @staticmethod
    def plot_similarities(t_range, data, vocab, keys=False, autoscale=False, title='Similarity', sort_legend=True, subplot_nrows=0, subplot_ncols=0, subplot_i = 1):
        pass

    def plot(self, sim, p):
        pass
        


    def evaluate(self, p, sim, plt):
        sim.run(p.xp.T)
        data = self.evaluate_behaviour(sim, p, p.seed)
        if plt:
            self.plot(sim, p)

        return data

class Model():
    def __init__(
        self, 
        vocabs, 
        experiment, 
        n_samples, 
        integrator_reset,
        proc_feedback, 
        compare_tau, 
        BG_thr, 
        BG_bias, 
        n_neurons_per_dim, 
        seed=None, 
        plot=False
    ):
        self.vocabs = vocabs
        self.experiment = experiment
        self.n_samples = n_samples
        self.integrator_reset = integrator_reset
        self.proc_feedback = proc_feedback
        self.compare_tau = compare_tau
        self.BG_thr = BG_thr
        self.BG_bias = BG_bias
        self.n_neurons_per_dim = n_neurons_per_dim
        self.plot = plot
        self.set_seed(seed) 

        self.network = spa.Network(seed=self.seed)
        self.construct_network()


    def send_n_neurons_per_dim(self):
        with self.network:
            self.network.config[spa.State].neurons_per_dimension = self.n_neurons_per_dim['State']
            self.network.config[spa.Bind].neurons_per_dimension = self.n_neurons_per_dim['Bind']
            self.network.config[spa.BasalGanglia].n_neurons_per_ensemble = self.n_neurons_per_dim['BG']
            self.network.config[spa.Thalamus].neurons_action = self.n_neurons_per_dim['thal_1']
            self.network.config[spa.Thalamus].neurons_channel_dim = self.n_neurons_per_dim['thal_1']
            self.network.config[spa.Thalamus].neurons_gate = self.n_neurons_per_dim['thal_2']
            self.network.config[spa.Scalar].n_neurons = self.n_neurons_per_dim['Scalar']


    def construct_network(self):
        self.send_n_neurons_per_dim()
        s = spa.sym
        net = self.network

        # temporary parameters
        crosstalk = True
        crosstalk_lr = 1e-11

        with net:

            net.input_net = spa.Network(label='inputs', seed=self.seed)

            with net.input_net:
                net.input_net.INSTRUCTIONS = spa.Transcode(self.experiment.INSTRUCTIONS_input, input_vocab=self.vocabs['big vocab'], output_vocab=self.vocabs['big vocab'], label='INSTRUCTIONS')
                # net.input_net.GET_INSTRUCTIONS = spa.Transcode(self.experiment.GET_INSTRUCTIONS_input, input_vocab=self.vocabs['big vocab'], output_vocab=self.vocabs['big vocab'], label='GET INSTRUCTIONS')
                # net.input_net.SET_INSTRUCTIONS = spa.Transcode(self.experiment.SET_INSTRUCTIONS_input, input_vocab=self.vocabs['big vocab'], output_vocab=self.vocabs['big vocab'], label='SET INSTRUCTIONS')
            
            with net.input_net:
                net.input_net.RETINA_input = spa.Transcode(self.experiment.RETINA_input, input_vocab=self.vocabs['big vocab'], output_vocab=self.vocabs['big vocab'])

            net.V = DirectProcessor(
                self.vocabs['big vocab'], self.vocabs['big vocab'], 'V', 
                receiver=False, # V only sends info to GW
                seed=self.seed,
                prediction_out=crosstalk,
                n_neurons_per_dim=self.n_neurons_per_dim
            )
            nengo.Connection(net.input_net.RETINA_input.output, net.V.input.input, synapse=None)

            net.FIXATE_detector = nengo.Node(size_in=1, label='FIXATE detector')
            nengo.Connection(net.V.input.output, net.FIXATE_detector, transform=net.V.output_vocab.parse('FIXATE').v[None,:], synapse=.05) # some synaptic delay is necessary to ensure the stimulus has time to enter GW
            net.declare_output(net.FIXATE_detector, None)


            net.M = DirectProcessor(
                self.vocabs['big vocab'], self.vocabs['big vocab'], 'M',
                sender=False, # M only receives info from GW
                seed=self.seed,
                n_neurons_per_dim=self.n_neurons_per_dim
            )
                    
            net.BTN = nengo.Node(Button(
                SP_vectors=[self.vocabs['big vocab'].parse('LESS').v, self.vocabs['big vocab'].parse('MORE').v], 
                trial_length=self.experiment.trial_length,
                wait_length=self.experiment.t_start), 
                label='button',
                size_in=self.vocabs['big vocab'].dimensions)
            nengo.Connection(net.M.output.output, net.BTN) # Connect output of M to BTN that records behavioral response

            # net.ADD = ADDProcessor(
            #     self.vocabs['big vocab'],
            #     'ADD',
            #     self.n_neurons_per_dim,
            #     self.rng,
            #     self.BG_bias,
            #     self.BG_thr,
            #     feedback=self.proc_feedback,
            #     seed=self.seed
            # )
            # with net.input_net:
            #     net.input_net.ADDEND = spa.Transcode(self.experiment.ADDEND_input, input_vocab=self.vocabs['big vocab'], output_vocab=self.vocabs['big vocab'], label='ADDEND')
            # nengo.Connection(net.input_net.ADDEND.output, net.ADD.bind.input_right, synapse=None)
            
            if self.compare_tau != 0:
                net.COM = CompareProcessor(
                    self.vocabs['big vocab'], 
                    self.vocabs['big vocab'], 
                    'COM',
                    self.experiment.trial_length,
                    self.experiment.t_start if self.integrator_reset else 0,
                    self.n_neurons_per_dim,
                    self.rng,
                    self.n_samples,
                    tau=self.compare_tau,
                    seed=self.seed,
                    prediction_in=crosstalk
                    )
            else:
                net.COM = AMProcessor(
                    self.vocabs['big vocab'], self.vocabs['big vocab'], 'COM',
                    {   'D2':'LESS',
                        'D4':'LESS',
                        'D6':'MORE',
                        'D8':'MORE'  },
                    n_neurons_per_dim=self.n_neurons_per_dim, seed=self.seed,
                    prediction_in=crosstalk
                )
            
            self.processors = [net.V, net.COM, net.M]#, net.ADD]

            if crosstalk:
                net.crosstalk = Prediction(
                    net.V, 
                    net.COM,
                    rate=crosstalk_lr
                )

            net.GW = GlobalWorkspace(
                self.vocabs['big vocab'],
                mappings={
                    net.V: ['D2','D4','D6','D8','FIXATE'],
                    # net.ADD: ['D2','D4','D6','D8'],
                    net.COM: ['MORE','LESS'],
                },
                n_neurons = self.n_neurons_per_dim['AM'],
                seed=self.seed
            )
            for detector in net.GW.detectors.values():
                net.declare_output(detector,None)

            net.POS = WM(100, self.vocabs['big vocab'])
            net.clean_POS = spa.WTAAssocMem(
                threshold=.2,
                input_vocab=net.POS.vocab,
                mapping=['D1','D2','D3','D4'],
                n_neurons=self.n_neurons_per_dim['AM'],
                function=lambda x: x>0
            )
            nengo.Connection(net.POS.output, net.clean_POS.input)
            net.INCREMENT = WM(100, self.vocabs['big vocab'])

            net.PRIM = spa.Bind(neurons_per_dimension=self.n_neurons_per_dim['Bind'], vocab=self.vocabs['big vocab'], unbind_right=True)
            net.GET_PRIM = spa.WTAAssocMem(
                threshold=.5,
                input_vocab=net.PRIM.vocab,
                mapping=['GET_V', 'GET_COM', 'GET_ADD'],
                n_neurons=self.n_neurons_per_dim['AM'],
                function=lambda x: x>0
            )
            net.SET_PRIM = spa.WTAAssocMem(
                threshold=.5,
                input_vocab=net.PRIM.vocab,
                mapping=['SET_COM', 'SET_ADD', 'SET_M'],
                n_neurons=self.n_neurons_per_dim['AM'],
                function=lambda x: x>0
            )
            net.PRIM >> net.GET_PRIM
            net.PRIM >> net.SET_PRIM

            net.input_net.INSTRUCTIONS >> net.PRIM.input_left
            spa.translate(net.clean_POS, self.vocabs['big vocab']) >> net.PRIM.input_right

            # GET selector
            with spa.Network(label='GET selector', seed=self.seed) as net.GET_selector:
                net.GET_selector.labels = []
                with spa.ActionSelection() as net.GET_selector.AS:
                    
                    net.GET_selector.labels.append("GET V (FIXATE)")
                    spa.ifmax(net.GET_selector.labels[-1], self.BG_bias + net.FIXATE_detector,
                        net.V.preconscious >> net.GW.AMs[net.V].input,
                        s.D1 >> net.POS.input,
                        s.D1 * net.clean_POS >> net.INCREMENT.input
                    )

                    # net.GET_selector.labels.append("GET V")
                    # spa.ifmax(net.GET_selector.labels[-1], self.BG_bias + spa.dot(net.GET_PRIM, s.GET_V) * (1-net.GW.detectors[net.V]),
                    #     net.V.preconscious >> net.GW.AMs[net.V].input,
                    #     1 >> net.POS.gate,
                    # )

                    # net.GET_selector.labels.append("GET ADD")
                    # spa.ifmax(net.GET_selector.labels[-1], self.BG_bias + spa.dot(net.GET_PRIM, s.GET_ADD) * (1-net.GW.detectors[net.ADD]),
                    #     net.ADD.preconscious >> net.GW.AMs[net.ADD].input,
                    #     1 >> net.POS.gate,
                    #     s.D1 * net.clean_POS >> net.INCREMENT.input
                    # )
                    
                    net.GET_selector.labels.append("GET COM")
                    spa.ifmax(net.GET_selector.labels[-1], self.BG_bias + spa.dot(net.GET_PRIM, s.GET_COM) * (1-net.GW.detectors[net.COM]),
                        net.COM.preconscious >> net.GW.AMs[net.COM].input,
                        1 >> net.POS.gate,
                        s.D1 * net.clean_POS >> net.INCREMENT.input
                    )

                    net.GET_selector.labels.append("Thresholder")
                    spa.ifmax(net.GET_selector.labels[-1], self.BG_bias + self.BG_thr,
                        1 >> net.INCREMENT.gate,
                        net.INCREMENT >> net.POS.input

                    )

            # SET selector
            with spa.Network(label='SET selector', seed=self.seed) as net.SET_selector:
                net.SET_selector.labels = []
                with spa.ActionSelection() as net.SET_selector.AS:

                    # net.SET_selector.labels.append("SET ADD")
                    # spa.ifmax(net.SET_selector.labels[-1], self.BG_bias + spa.dot(net.SET_PRIM, s.SET_ADD) * (1-net.GW.detectors[net.ADD]),
                    #     net.GW.AMs[net.COM] >> net.ADD.broadcast,
                    #     net.GW.AMs[net.V] >> net.ADD.broadcast,
                    # )
                    
                    net.SET_selector.labels.append("SET COM")
                    spa.ifmax(net.SET_selector.labels[-1], self.BG_bias + spa.dot(net.SET_PRIM, s.SET_COM) * (1-net.GW.detectors[net.COM]),
                        # net.GW.AMs[net.ADD] >> net.COM.broadcast,
                        net.GW.AMs[net.V] >> net.COM.broadcast,
                    )

                    net.SET_selector.labels.append("SET M")
                    spa.ifmax(net.SET_selector.labels[-1], self.BG_bias + spa.dot(net.SET_PRIM, s.SET_M),
                        net.GW.AMs[net.COM] >> net.M.broadcast,
                        net.GW.AMs[net.V] >> net.M.broadcast,
                        # net.GW.AMs[net.ADD] >> net.M.broadcast,
                    )

                    net.SET_selector.labels.append("Thresholder")
                    spa.ifmax(net.SET_selector.labels[-1], self.BG_bias + self.BG_thr) # Threshold for action




    @property
    def senders(self):
        return [p for p in self.processors if p.sender]

    @property
    def receivers(self):
        return [p for p in self.processors if p.receiver]

    def set_seed(self, seed=None):
        if seed is None:
            self.seed = np.random.randint(999) # random seed
            print("Warning: setting random seed")
        else:
            self.seed = seed

        self.send_seed()

    def send_seed(self):
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.rng = np.random.RandomState(self.seed)

    def make_probes(self, synapse=.005):
        net = self.network
        with net:

            self.probes = {'BTN': nengo.Probe(net.BTN, synapse=None)}

            if self.plot:
                # Processors
                self.probes.update({p: 
                    {
                        'in': nengo.Probe(p.processing_input.output, synapse=synapse),
                        'out': nengo.Probe(p.output.output, synapse=synapse)
                    }
                    for p in self.processors})

                for p in self.senders:
                    self.probes[p]['preconscious'] = nengo.Probe(p.preconscious.output, synapse=synapse)

                # for p in self.receivers:
                #     self.probes[p]['broadcast'] = nengo.Probe(p.broadcast.output, synapse=synapse)
                
                if self.compare_tau != 0:
                    self.probes[net.COM]['compared'] = nengo.Probe(net.COM.compared, synapse=synapse)
                    self.probes[net.COM]['integrator'] = nengo.Probe(net.COM.integrator, synapse=synapse)

                # GW
                self.probes.update({net.GW: {
                    'in': {p: nengo.Probe(net.GW.AMs[p].input, synapse=synapse) for p in self.senders},
                    'out': nengo.Probe(net.GW.output, synapse=synapse),
                    'voltages': [nengo.Probe(ens.neurons, 'voltage', synapse=synapse) for ens in net.GW.all_ensembles]}})
                

                # instruction-following system
                self.probes.update({net.POS: nengo.Probe(net.POS.output, synapse=synapse)})
                self.probes.update({net.INCREMENT: nengo.Probe(net.INCREMENT.output, synapse=synapse)})
                self.probes.update({net.GET_PRIM: nengo.Probe(net.GET_PRIM.output, synapse=synapse)})
                self.probes.update({net.SET_PRIM: nengo.Probe(net.SET_PRIM.output, synapse=synapse)})
                        
                # Action selection networks
                self.probes.update({AS_net: {'in': nengo.Probe(AS_net.AS.bg.input, synapse=synapse),
                                                          'out': nengo.Probe(AS_net.AS.thalamus.output, synapse=synapse)}
                                    for AS_net in [net.GET_selector, net.SET_selector]})#, net.ADD.result_controller]})#, net.access_net] + net.broadcast_nets}})

    def run(self, simulator_cls, dt=.001):
        # print("Number of neurons:", self.network.n_neurons)
        # print("T:", self.experiment.trial_length*len(self.experiment.trials))

        self.send_seed()
        self.make_probes()
        with simulator_cls(self.network, dt=dt, seed=self.seed) as sim:
            sim.run(self.experiment.trial_length*len(self.experiment.trials))

        return sim

