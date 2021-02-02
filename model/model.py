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
        self.crosstalk = False
        self.crosstalk_lr = 5e-12

        with net:

            net.input_net = spa.Network(label='inputs', seed=self.seed)

            with net.input_net:
                net.input_net.RETINA_input = spa.Transcode(self.experiment.RETINA_input, input_vocab=self.vocabs['GW'], output_vocab=self.vocabs['GW'])

            net.V = DirectProcessor(
                self.vocabs['GW'], self.vocabs['GW'], 'V', 
                receiver=False, # V only sends info to GW
                seed=self.seed,
                prediction_out=self.crosstalk,
                n_neurons_per_dim=self.n_neurons_per_dim
            )
            nengo.Connection(net.input_net.RETINA_input.output, net.V.input.input, synapse=None)

            net.FIXATE_detector = nengo.Node(size_in=1, label='FIXATE detector')
            nengo.Connection(net.V.input.output, net.FIXATE_detector, transform=net.V.output_vocab.parse('FIXATE').v[None,:], synapse=.05) # some synaptic delay is necessary to ensure the stimulus has time to enter GW
            net.declare_output(net.FIXATE_detector, None)


            net.M = DirectProcessor(
                self.vocabs['GW'], self.vocabs['GW'], 'M',
                sender=False, # M only receives info from GW
                seed=self.seed,
                n_neurons_per_dim=self.n_neurons_per_dim
            )
                    
            net.BTN = nengo.Node(Button(
                SP_vectors=[self.vocabs['GW'].parse('LESS').v, self.vocabs['GW'].parse('MORE').v], 
                trial_length=self.experiment.trial_length,
                wait_length=self.experiment.t_start), 
                label='button',
                size_in=self.vocabs['GW'].dimensions)
            nengo.Connection(net.M.output.output, net.BTN) # Connect output of M to BTN that records behavioral response

            net.ADD = ADDProcessor(
                self.vocabs['GW'],
                'ADD',
                self.n_neurons_per_dim,
                self.rng,
                self.BG_bias,
                self.BG_thr,
                feedback=self.proc_feedback,
                seed=self.seed
            )
            with net.input_net:
                net.input_net.ADDEND = spa.Transcode(self.experiment.ADDEND_input, input_vocab=self.vocabs['GW'], output_vocab=self.vocabs['GW'], label='ADDEND')
            nengo.Connection(net.input_net.ADDEND.output, net.ADD.bind.input_right, synapse=None)
            
            if self.compare_tau != 0:
                net.COM = CompareProcessor(
                    self.vocabs['GW'], 
                    self.vocabs['GW'], 
                    'COM',
                    self.experiment.trial_length,
                    self.experiment.t_start if self.integrator_reset else 0,
                    self.n_neurons_per_dim,
                    self.rng,
                    self.n_samples,
                    tau=self.compare_tau,
                    seed=self.seed,
                    prediction_in=self.crosstalk
                    )
            else:
                net.COM = AMProcessor(
                    self.vocabs['GW'], self.vocabs['GW'], 'COM',
                    {   'D2':'LESS',
                        'D4':'LESS',
                        'D6':'MORE',
                        'D8':'MORE'  },
                    n_neurons_per_dim=self.n_neurons_per_dim, seed=self.seed,
                    prediction_in=self.crosstalk
                )
            
            self.processors = [net.V, net.COM, net.M, net.ADD]

            if self.crosstalk:
                net.crosstalk = Prediction(
                    net.V, 
                    net.COM,
                    rate=self.crosstalk_lr
                )

            net.GW = GlobalWorkspace(
                self.vocabs['GW'],
                mappings={
                    net.V: ['D2','D4','D6','D8','FIXATE'],
                    net.ADD: ['D2','D4','D6','D8'],
                    net.COM: ['MORE','LESS'],
                },
                n_neurons = self.n_neurons_per_dim['AM'],
                seed=self.seed
            )
            for detector in net.GW.detectors.values():
                net.declare_output(detector,None)

            net.POS = WM(200, self.vocabs['PRIM'])
            net.clean_POS = spa.WTAAssocMem(
                threshold=0,
                input_vocab=net.POS.vocab,
                mapping=['D1','D2','D3','D4'],
                n_neurons=self.n_neurons_per_dim['AM'],
                function=lambda x: x>0
            )
            nengo.Connection(net.POS.output, net.clean_POS.input)
            
            net.INCREMENT = WM(200, self.vocabs['PRIM'])

            nengo.Connection(net.clean_POS.output, net.INCREMENT.input, transform=net.POS.vocab.parse('D1').get_binding_matrix())
            nengo.Connection(net.INCREMENT.output, net.POS.input)

            with net.input_net:
                net.input_net.INSTRUCTIONS = spa.Transcode(self.experiment.INSTRUCTIONS_input, input_vocab=self.vocabs['PRIM'], output_vocab=self.vocabs['PRIM'], label='INSTRUCTIONS')
            
            net.PRIM = spa.Bind(neurons_per_dimension=self.n_neurons_per_dim['Bind'], vocab=self.vocabs['PRIM'], unbind_right=True)

            net.input_net.INSTRUCTIONS >> net.PRIM.input_left
            net.clean_POS >> net.PRIM.input_right

            # GET selector
            with spa.Network(label='GET selector', seed=self.seed) as net.GET_selector:
                net.GET_selector.labels = []
                with spa.ActionSelection() as net.GET_selector.AS:
                    
                    net.GET_selector.labels.append("GET V (FIXATE)")
                    spa.ifmax(net.GET_selector.labels[-1], self.BG_bias + net.FIXATE_detector,
                        net.V.preconscious >> net.GW.AMs[net.V].input,
                        s.D1 >> net.POS.input,
                        1 >> net.INCREMENT.reset
                    )

                    net.GET_selector.labels.append("GET ADD")
                    spa.ifmax(net.GET_selector.labels[-1], self.BG_bias + spa.dot(net.PRIM, s.GET_ADD) * (1-net.GW.detectors[net.ADD]),
                        net.ADD.preconscious >> net.GW.AMs[net.ADD].input,
                        1 >> net.POS.gate,
                    )
                    
                    net.GET_selector.labels.append("GET COM")
                    spa.ifmax(net.GET_selector.labels[-1], self.BG_bias + spa.dot(net.PRIM, s.GET_COM) * (1-net.GW.detectors[net.COM]),
                        net.COM.preconscious >> net.GW.AMs[net.COM].input,
                        1 >> net.POS.gate,
                    )

                    net.GET_selector.labels.append("Thresholder")
                    spa.ifmax(net.GET_selector.labels[-1], self.BG_bias + self.BG_thr,
                        1 >> net.INCREMENT.gate,
                    )

            # SET selector
            with spa.Network(label='SET selector', seed=self.seed) as net.SET_selector:
                net.SET_selector.labels = []
                with spa.ActionSelection() as net.SET_selector.AS:

                    net.SET_selector.labels.append("SET ADD")
                    spa.ifmax(net.SET_selector.labels[-1], self.BG_bias + spa.dot(net.PRIM, s.SET_ADD) * (1-net.GW.detectors[net.ADD]),
                        net.GW.AMs[net.COM] >> net.ADD.broadcast,
                        net.GW.AMs[net.V] >> net.ADD.broadcast,
                    )
                    
                    net.SET_selector.labels.append("SET COM")
                    spa.ifmax(net.SET_selector.labels[-1], self.BG_bias + spa.dot(net.PRIM, s.SET_COM) * (1-net.GW.detectors[net.COM]),
                        net.GW.AMs[net.ADD] >> net.COM.broadcast,
                        net.GW.AMs[net.V] >> net.COM.broadcast,
                    )

                    net.SET_selector.labels.append("SET M")
                    spa.ifmax(net.SET_selector.labels[-1], self.BG_bias + spa.dot(net.PRIM, s.SET_M),
                        net.GW.AMs[net.COM] >> net.M.broadcast,
                        net.GW.AMs[net.V] >> net.M.broadcast,
                        net.GW.AMs[net.ADD] >> net.M.broadcast,
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

                for p in self.processors:
                    if p.prediction_in:
                        self.probes[p]['prediction in'] = nengo.Probe(p.prediction_in_ens, synapse=synapse)
                    if p.prediction_out:
                        self.probes[p]['prediction out'] = nengo.Probe(p.prediction_out_ens, synapse=synapse)

                self.probes[net.ADD]['addend'] = nengo.Probe(net.ADD.bind.input_right, synapse=synapse)

                if self.compare_tau != 0:
                    self.probes[net.COM]['compared'] = nengo.Probe(net.COM.compared, synapse=synapse)
                    self.probes[net.COM]['integrator'] = nengo.Probe(net.COM.integrator, synapse=synapse)

                # GW
                self.probes.update({net.GW: {
                    'in': {p: nengo.Probe(net.GW.AMs[p].input, synapse=synapse) for p in self.senders},
                    'out': nengo.Probe(net.GW.output, synapse=synapse),
                    'voltages': [nengo.Probe(ens.neurons, 'voltage', synapse=synapse) for ens in net.GW.all_ensembles]}})
                

                # instruction-following system
                self.probes.update({wm: 
                    {
                        'in': nengo.Probe(wm.input, synapse=synapse),
                        'gate': nengo.Probe(wm.gate, synapse=synapse),
                        'out': nengo.Probe(wm.output, synapse=synapse)
                    }
                    for wm in [net.POS, net.INCREMENT]})
                self.probes.update({clean: nengo.Probe(clean.output, synapse=synapse) for clean in [net.clean_POS]})
                self.probes.update({net.PRIM: nengo.Probe(net.PRIM.output, synapse=synapse)})
                        
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

