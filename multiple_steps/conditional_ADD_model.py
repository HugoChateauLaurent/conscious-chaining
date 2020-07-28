import nengo
import nengo_spa as spa
import numpy as np
from modules import AMProcessor, Button, Delay
from data import Data, SackurData
import random
import pandas as pd
import matplotlib.pyplot as plt

import pytry

def create_vocab(D, seed):
	pointers = ['D'+str(i) for i in range(2,8+1)]
	pointers += [
		'FIXATE', #'MASK',
		'MORE','LESS',
		'GET','SET',
		'SIMPLE', 'CHAINED_ADD', 'CHAINED_SUB',
		'ON',
		'V', 'COM', 'ADD', 'SUB', 'M'
	]
	rng = np.random.RandomState(seed)
	vocab = spa.Vocabulary(dimensions=D, pointer_gen=rng)
	rng.shuffle(pointers) # avoid bias in similarity
	vocab.populate(";".join(pointers)) # add pointers to vocabuary
	return vocab

class ExperimentRun(pytry.NengoTrial):
	def params(self):
		self.param('scaling of the number of neurons', n_neurons_scale=1)
		self.param('vocab', vocab=None)
		self.param('experiment', xp=None)
		self.param('Number of addition processors', n_ADD_proc=1)
		self.param('Number of steps', N_steps=1)
		self.param('feedback of processors', proc_feedback=0)
		self.param('feedback synapse of processors', proc_feedback_synapse=.1)
		self.param('feedback of global workspace', GW_feedback=0)
		self.param('threshold for global workspace WTA networks', GW_threshold=.5)
		self.param('amplification of global workspace input signal', GW_scale=20)
		self.param('threshold for action selection networks', BG_thr=.1)
		self.param('bias for action selection networks', BG_bias=.5)

	def make_n_neurons_per_dim(self, p):
		# Number of neurons
		n_neurons_per_dim = { # initialize to default values then scale
			module: int(number*p.n_neurons_scale) for module,number in { 
			'AM': 50,
			'Ensemble': 50,
			'State': 50,
			'BG': 100,
			'thal_1': 50,
			'thal_2': 40,
			'Scalar': 50 # this is for the whole module (one dim)
		}.items()}

		return n_neurons_per_dim

	def model(self, p):
		n_neurons_per_dim = self.make_n_neurons_per_dim(p)
		model = Model(p.vocab, p.xp, p.n_ADD_proc, p.N_steps, p.proc_feedback, p.proc_feedback_synapse, p.GW_feedback, p.GW_threshold, p.GW_scale, p.BG_thr, p.BG_bias, n_neurons_per_dim, p.seed, p.plt)
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

	def evaluate(self, p, sim, plt):
		sim.run(p.xp.T)
		data = self.evaluate_behaviour(sim, p, p.seed)
		
		return data

class Model():
	def __init__(self, vocab, experiment, proc_feedback, proc_feedback_synapse,
			GW_feedback, GW_threshold, GW_scale, BG_thr, BG_bias, n_neurons_per_dim, seed=None, plot=False):
		self.vocab = vocab
		self.experiment = experiment
		self.proc_feedback = proc_feedback
		self.proc_feedback_synapse = proc_feedback_synapse
		self.GW_feedback = GW_feedback
		self.GW_threshold = GW_threshold
		self.GW_scale = GW_scale
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
			self.network.config[spa.BasalGanglia].n_neurons_per_ensemble = self.n_neurons_per_dim['BG']
			self.network.config[spa.Thalamus].neurons_action = self.n_neurons_per_dim['thal_1']
			self.network.config[spa.Thalamus].neurons_channel_dim = self.n_neurons_per_dim['thal_1']
			self.network.config[spa.Thalamus].neurons_gate = self.n_neurons_per_dim['thal_2']
			self.network.config[spa.Scalar].n_neurons = self.n_neurons_per_dim['Scalar']


	def construct_network(self):
		self.send_n_neurons_per_dim()
		s = spa.sym
		net = self.network
		with net:

			net.input_net = spa.Network(label='inputs', seed=self.seed)

			# We start defining the buffer slots in which information can
			# be placed:
			
			# A slot for the visual input (the digit N). Feedback is used for iconic memory (100-300ms)
			V_mapping = ['D2','D4','D6','D8','FIXATE']#,'MASK']
			with net.input_net:
				net.input_net.RETINA_input = spa.Transcode(self.experiment.RETINA_input, output_vocab=self.vocab)
				net.input_net.senso_delay = nengo.Node(Delay(self.vocab.dimensions, self.t_senso).step, size_in=self.vocab.dimensions, size_out=self.vocab.dimensions) # sensory processing delay
				nengo.Connection(net.input_net.RETINA_input.output, net.input_net.senso_delay, synapse=None)

			net.V = AMProcessor(
				self.vocab, self.vocab, 'V', 
				V_mapping, 
				feedback=self.proc_feedback,
				feedback_synapse=self.proc_feedback_synapse,
				receiver=False, # V only sends info to GW
				npd_AM=self.n_neurons_per_dim['AM'], seed=self.seed)
			nengo.Connection(net.input_net.senso_delay, net.V.input.input, synapse=None)

			# A slot for the action (MORE or LESS)
			net.M = AMProcessor(
				self.vocab, self.vocab, 'M',
				['MORE','LESS'],
				feedback=0,#self.proc_feedback,
				feedback_synapse=0,#self.proc_feedback_synapse,
				sender=False, # M only receives info from GW
				npd_AM=self.n_neurons_per_dim['AM'], seed=self.seed)
					
			net.BTN = nengo.Node(Button(
				SP_vectors=[self.vocab.parse('LESS').v, self.vocab.parse('MORE').v], 
				trial_length=self.experiment.trial_length,
				wait_length=self.experiment.t_start), 
				size_in=self.vocab.dimensions)
			nengo.Connection(net.M.preconscious.output, net.BTN, synapse=None) # Connect output of M to BTN that records behavioral response

			
			self.ADD_dict = {}
			for p_i in range(self.n_ADD_proc):
				self.ADD_dict['P'+str(p_i)] = AMProcessor(
					self.vocab, self.vocab, 'P'+str(p_i),
					{'D'+str(self.)},
					feedback=self.proc_feedback,
					feedback_synapse=self.proc_feedback_synapse,
					npd_AM=self.n_neurons_per_dim['AM'], seed=self.seed
				)
			
			
			self.processors = [net.V, net.M] + list(self.ADD_dict.values())
			net.PREV = spa.State(self.vocab, feedback=1, feedback_synapse=.005, label='PREV')
			net.COUNT = spa.State(self.vocab, feedback=1, feedback_synapse=.005, label='COUNT')
			   
			# Selects information from the processors
			net.GW = spa.WTAAssocMem(
				threshold=self.GW_threshold, 
				input_vocab=self.vocab, 
				output_vocab=self.vocab,
				mapping=['D2','D4','D6','D8', 'FIXATE', 'MORE','LESS'],#,'MASK'],
				function=lambda x:x>0,
				label='GW content',
				n_neurons = self.n_neurons_per_dim['AM']
			)
			nengo.Connection(net.GW.selection.output, net.GW.selection.input,  # feedback
				transform=self.GW_feedback, synapse=.02)
			for p in self.senders:
				nengo.Connection(p.attention_weighting.output, net.GW.input, synapse=None)
			for p in self.receivers:
				nengo.Connection(net.GW.output, p.broadcast.output, synapse=None)

			# routing network
			with spa.Network(label='routing', seed=self.seed) as net.routing_net:
				net.routing_net.labels = []
				with spa.ActionSelection() as net.routing_net.AS:
					
					net.routing_net.labels.append("GET V")
					spa.ifmax(net.routing_net.labels[-1],  self.BG_bias + spa.dot(net.V.preconscious, s.FIXATE) + spa.dot(net.GW, s.FIXATE),
								  self.GW_scale >> net.V.attention,
								  .25*s.GET*s.V >> net.PREV
							 )

					for p_i in range(1, self.n_ADD_proc-1):
						net.routing_net.labels.append("SET P"+str(p_i))
						spa.ifmax(net.routing_net.labels[-1], self.BG_bias
						+	spa.dot(net.PREV, s.GET*self.vocab.parse('P'+str(p_i-1)))
						- 	spa.dot(net.GW, s.FIXATE) 	# too soon
						,
									  1 >> net.ADD_dict['P'+str(p_i)].receive,
									  .25*s.SET*self.vocab.parse('P'+str(p_i)) >> net.PREV,
								 )

						net.routing_net.labels.append("GET P"+str(p_i))
						spa.ifmax(net.routing_net.labels[-1], self.BG_bias
						+ 	spa.dot(net.PREV, s.SET*self.vocab.parse('P'+str(p_i))) * spa.dot(net.ADD_dict['P'+str(p_i)].preconscious, s.ON),
									  self.GW_scale >> net.ADD_dict['P'+str(p_i)].attention,
									  .25*s.GET*self.vocab.parse('P'+str(p_i)) >> net.PREV,
								 )

					net.routing_net.labels.append("SET M")
					spa.ifmax(net.routing_net.labels[-1], self.BG_bias 
					+ 	spa.dot(net.PREV, s.GET*s.COM) * spa.dot(net.COUNT, s.COUNT*vocab.parse(")
					+ 	.5*spa.dot(net.PREV, s.SET*s.M) # sustain
					,
								  1 >> net.M.receive,
								  .25*s.SET*s.M >> net.PREV,
							 )

					net.routing_net.labels.append("Thresholder")
					spa.ifmax(net.routing_net.labels[-1], self.BG_bias + self.BG_thr) # Threshold for action




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

	def make_probes(self, synapse=.015):
		net = self.network
		with net:

			self.probes = {'BTN': nengo.Probe(net.BTN)}

			if self.plot:
				# Processors
				self.probes.update({'processors': {p.label: {'in': nengo.Probe(p.input.input, synapse=synapse),
														# 'out': nengo.Probe(p.AM.output, synapse=synapse) if p != net.COM else nengo.Probe(p.preconscious, synapse=synapse)}
														'out': nengo.Probe(p.preconscious.output, synapse=synapse)}
									for p in self.processors}})

				for p in self.senders:
					self.probes['processors'][p.label]['attention weighted'] = nengo.Probe(p.attention_weighting.output, synapse=synapse)
				
				for p in self.receivers:
					self.probes['processors'][p.label]['broadcast'] = nengo.Probe(p.broadcast.output, synapse=synapse)
				
				if self.s_evidence is not None:
					self.probes['processors']['COM']['clean input'] = nengo.Probe(net.COM.clean_a.output, synapse=synapse)
					self.probes.update({'compared': nengo.Probe(net.COM.compared, synapse=synapse)})
					self.probes.update({'integrator': nengo.Probe(net.COM.integrator, synapse=synapse)})

				# GW
				self.probes.update({'GW': { 'in': nengo.Probe(net.GW.input, synapse=synapse),
											'out': nengo.Probe(net.GW.output, synapse=synapse)}
							})
				# self.probes.update({'GW source': { 'in': nengo.Probe(net.GW_source.input, synapse=synapse),
				# 							'out': nengo.Probe(net.GW_source.output, synapse=synapse)}
				# 			})

				# for p in self.senders:
				# 	self.probes['processors'][p.label]['sent'] = nengo.Probe(p.sent.output, synapse=None)
				# for p in self.receivers:
				# 	self.probes['processors'][p.label]['received'] = nengo.Probe(p.received.output, synapse=None)

				# PREV and G
				self.probes.update({state.label: nengo.Probe(state.input, synapse=synapse)
							for state in [net.PREV, net.G]})
						
				# Action selection networks
				self.probes.update({'AS_nets': {AS_net.label: {'in': nengo.Probe(AS_net.AS.bg.input, synapse=synapse),
														  'out': nengo.Probe(AS_net.AS.thalamus.output, synapse=synapse)}
									for AS_net in [net.routing_net]}})#, net.access_net] + net.broadcast_nets}})

				# Attentional levels
				self.probes.update({'attention': {p.label+" attention": nengo.Probe(p.attention.output, synapse=synapse)
									for p in self.senders}})

				# Receive levels
				self.probes.update({'receive': {p.label+" receive": nengo.Probe(p.receive.output, synapse=synapse)
									for p in self.receivers}})

	def run(self, simulator_cls, dt=.001):
		# print("Number of neurons:", self.network.n_neurons)
		# print("T:", self.experiment.trial_length*len(self.experiment.trials))

		self.send_seed()
		self.make_probes()
		with simulator_cls(self.network, dt=dt, seed=self.seed) as sim:
			sim.run(self.experiment.trial_length*len(self.experiment.trials))

		return sim

