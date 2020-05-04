import nengo
import nengo_spa as spa
import numpy as np
from modules import Processor, Button
import random

class Model():
	def __init__(self, vocab, experiment, proc_feedback, GW_feedback, GW_scale, BG_thr, BG_bias, n_neurons_per_dim, seed=None):
		self.vocab = vocab
		self.experiment = experiment
		self.proc_feedback = proc_feedback
		self.GW_feedback = GW_feedback
		self.GW_scale = GW_scale
		self.BG_thr = BG_thr
		self.BG_bias = BG_bias
		self.n_neurons_per_dim = n_neurons_per_dim
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
			
			# A slot for the goal/task
			net.G = spa.State(self.vocab, label='G')
			with net.input_net:
				net.input_net.G_input = spa.Transcode(self.experiment.G_input, output_vocab=self.vocab)
			net.input_net.G_input >> net.G
			
			# A slot for the visual input (the digit N). Feedback is used for iconic memory (100-300ms)
			V_mapping = {
				digit:'CONTENT*'+digit for digit in ['TWO','FOUR','SIX','EIGHT','FIXATE','MASK']}
			with net.input_net:
				net.input_net.RETINA_input = spa.Transcode(self.experiment.RETINA_input, output_vocab=self.vocab)
				
			net.V = Processor(
				self.vocab, self.vocab, 'V', 
				V_mapping, 
				feedback=self.proc_feedback,
				receiver=False, # V only sends info to GW
				npd_AM=self.n_neurons_per_dim['AM'], seed=self.seed)
			nengo.Connection(net.input_net.RETINA_input.output, net.V.input, synapse=None)
			
			# A slot for the action (MORE or LESS)
			net.M = Processor(
				self.vocab, self.vocab, 'M',
				{stim:stim+'*CONTENT' for stim in ['MORE','LESS']},
				feedback=self.proc_feedback,
				sender=False, # M only receives info from GW
				npd_AM=self.n_neurons_per_dim['AM'], seed=self.seed)
					
			net.BTN = nengo.Node(Button(
				[self.vocab.parse('CONTENT*LESS').v, self.vocab.parse('CONTENT*MORE').v], 
				self.experiment.trial_length), 
				size_in=self.vocab.dimensions)
			nengo.Connection(net.M.AM.output, net.BTN) # Connect output of M to BTN that records behavioral response

			# An associative memory for the + operation
			net.ADD = Processor(
				self.vocab, self.vocab, 'ADD',
				{   'TWO':'FOUR*CONTENT',
					'FOUR':'SIX*CONTENT',
					'SIX':'EIGHT*CONTENT',
					'EIGHT':'TWO*CONTENT'   },
				feedback=self.proc_feedback,
				npd_AM=self.n_neurons_per_dim['AM'], seed=self.seed
			)
			
			# An associative memory for the - operation
			net.SUB = Processor(
				self.vocab, self.vocab, 'SUB',
				{   'TWO':'EIGHT*CONTENT',
					'FOUR':'TWO*CONTENT',
					'SIX':'FOUR*CONTENT',
					'EIGHT':'SIX*CONTENT'   },
				feedback=self.proc_feedback,
				npd_AM=self.n_neurons_per_dim['AM'], seed=self.seed
			)
			
			# An associative memory for the "compare to 5" operation
			net.COM = Processor(
				self.vocab, self.vocab, 'COM',
				{   'TWO':'LESS*CONTENT',
					'FOUR':'LESS*CONTENT',
					'SIX':'MORE*CONTENT',
					'EIGHT':'MORE*CONTENT'  },
				feedback=self.proc_feedback,
				npd_AM=self.n_neurons_per_dim['AM'], seed=self.seed
			)
			
			self.processors = [net.V, net.ADD, net.SUB, net.COM, net.M]

			# Selects information from the processors
			net.GW_content = spa.WTAAssocMem(
				threshold=0, 
				input_vocab=self.vocab, 
				output_vocab=self.vocab,
				mapping=['TWO','FOUR','SIX','EIGHT', 'FIXATE','MASK', 'MORE','LESS'],
				function=lambda x:x>0,
				label='GW content',
				n_neurons = self.n_neurons_per_dim['AM']
			)
			for ens in net.GW_content.all_ensembles: # Add feedback to each ensemble
				nengo.Connection(ens, ens, transform=self.GW_feedback, synapse=.05)
			
			# Selects source of information (i.e. determines which processor sent the information)
			net.GW_source = spa.WTAAssocMem(
				threshold=0, 
				input_vocab=self.vocab, 
				output_vocab=self.vocab,
				mapping=[p.label for p in self.senders],
				function=lambda x:x>0,
				label='GW source',
				n_neurons = self.n_neurons_per_dim['AM']
			)
			for ens in net.GW_source.all_ensembles: # Add feedback to each ensemble
				nengo.Connection(ens, ens, transform=self.GW_feedback, synapse=.05)
			
			net.PREV = spa.ThresholdingAssocMem(
				threshold=0, 
				input_vocab=self.vocab, 
				output_vocab=self.vocab,
				mapping=['FIXATE','V','COM','ADD','SUB'],
				label='PREV',
				n_neurons = self.n_neurons_per_dim['AM']
			)
			for ens in net.PREV.all_ensembles: # Add feedback to each ensemble
				nengo.Connection(ens, ens, transform=.75, synapse=.05)
			
			# access network
			with spa.Network(label='access', seed=self.seed) as net.access_net:
				net.access_net.labels = []
				with spa.ActionSelection() as net.access_net.AS:
					for p in self.senders:
						net.access_net.labels.append(p.label)
						spa.ifmax(p.label, self.BG_bias+spa.dot(p.preconscious, s.SOURCE*self.vocab.parse(p.label)) * p.attention,
									  100 * self.GW_scale * p.preconscious * ~s.CONTENT >> net.GW_content.input,
									  self.GW_scale * self.vocab.parse(p.label) >> net.GW_source.input
									 )
					net.access_net.labels.append("Thresholder")
					spa.ifmax(self.BG_bias + self.BG_thr) 
			
			# broadcast networks
			net.broadcast_nets = []
			for p in self.receivers: # each processor p receives GW's content if p's "receive" level is more than a threshold
				net.broadcast_nets.append(spa.Network(label='broadcast '+p.label, seed=self.seed))
				net.broadcast_nets[-1].labels = []
				with net.broadcast_nets[-1]:
					with spa.ActionSelection() as net.broadcast_nets[-1].AS:
						net.broadcast_nets[-1].labels.append(p.label+" GO")
						spa.ifmax(net.broadcast_nets[-1].labels[-1], p.receive,
									 net.GW_content >> p.input
								 )
						net.broadcast_nets[-1].labels.append(p.label+" NOGO")
						spa.ifmax(net.broadcast_nets[-1].labels[-1], .5)
			   
			# routing network
			with spa.Network(label='routing', seed=self.seed) as net.routing_net:
				net.routing_net.labels = []
				with spa.ActionSelection() as net.routing_net.AS:
					
					net.routing_net.labels.append("ATTEND")
					spa.ifmax(net.routing_net.labels[-1],  self.BG_bias + spa.dot(net.GW_source, s.V) * spa.dot(net.GW_content, s.FIXATE),
								  s.FIXATE >> net.PREV,
								  *(.5 >> p.attention if p==net.V else -.5 >> p.attention for p in self.senders)
							 )
					
					net.routing_net.labels.append("V_COM")
					spa.ifmax(net.routing_net.labels[-1], self.BG_bias + spa.dot(net.GW_source, s.V) * spa.dot(net.G, s.SIMPLE) - spa.dot(net.GW_content, s.FIXATE) - spa.dot(net.GW_content, s.MASK) - spa.dot(net.PREV, s.V),
								  1 >> net.COM.receive,
								  s.V >> net.PREV,
								  *(.5 >> p.attention if p==net.COM else -.5 >> p.attention for p in self.senders)
							 )

					net.routing_net.labels.append("V_SUB")
					spa.ifmax(net.routing_net.labels[-1], self.BG_bias + spa.dot(net.GW_source, s.V) * spa.dot(net.G, s.CHAINED_SUB) - spa.dot(net.GW_content, s.FIXATE) - spa.dot(net.GW_content, s.MASK) - spa.dot(net.PREV, s.V),
								  1 >> net.SUB.receive,
								  s.V >> net.PREV,
								  *(.5 >> p.attention if p==net.SUB else -.5 >> p.attention for p in self.senders)
							 )

					net.routing_net.labels.append("V_ADD")
					spa.ifmax(net.routing_net.labels[-1], self.BG_bias + spa.dot(net.GW_source, s.V) * spa.dot(net.G, s.CHAINED_ADD) - spa.dot(net.GW_content, s.FIXATE) - spa.dot(net.GW_content, s.MASK) - spa.dot(net.PREV, s.V),
								  1 >> net.ADD.receive,
								  s.V >> net.PREV,
								  *(.5 >> p.attention if p==net.ADD else -.5 >> p.attention for p in self.senders)
							 )

					net.routing_net.labels.append("ADD_COM")
					spa.ifmax(net.routing_net.labels[-1], self.BG_bias + spa.dot(net.GW_source, s.ADD) * spa.dot(net.PREV, s.V),
								  1 >> net.COM.receive,
								  s.ADD >> net.PREV,
								  *(.5 >> p.attention if p==net.COM else -.5 >> p.attention for p in self.senders)
							 )

					net.routing_net.labels.append("SUB_COM")
					spa.ifmax(net.routing_net.labels[-1], self.BG_bias + spa.dot(net.GW_source, s.SUB) * spa.dot(net.PREV, s.V),
								  1 >> net.COM.receive,
								  s.SUB >> net.PREV,
								  *(.5 >> p.attention if p==net.COM else -.5 >> p.attention for p in self.senders)
							 )

					net.routing_net.labels.append("COM_M")
					spa.ifmax(net.routing_net.labels[-1], self.BG_bias + spa.dot(net.GW_source, s.COM) * (spa.dot(net.PREV, s.ADD) + spa.dot(net.PREV, s.SUB) + spa.dot(net.PREV, s.V)),
								  1 >> net.M.receive,
								  s.COM >> net.PREV,
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
		print("Seed:", self.seed)

		self.send_seed()

	def send_seed(self):
		np.random.seed(self.seed)
		random.seed(self.seed)

	def set_probes(self, synapse=.015):
		net = self.network
		with net:
			# Processors
			self.probes = {'processors': {p.label: {'in': nengo.Probe(p.input, synapse=synapse),
													'out': nengo.Probe(p.AM.output, synapse=synapse)}
								for p in self.processors}}
			
			self.probes.update({'BTN': nengo.Probe(net.BTN)})

			# GW
			self.probes.update({'GW': {GW.label: {  'in': nengo.Probe(GW.input, synapse=synapse),
													'out': nengo.Probe(GW.output, synapse=synapse)}
						for GW in [net.GW_source, net.GW_content]}})

			# PREV and G
			self.probes.update({state.label: nengo.Probe(state.input, synapse=synapse)
						for state in [net.PREV, net.G]})
					
			# Action selection networks
			self.probes.update({'AS_nets': {AS_net.label: {'in': nengo.Probe(AS_net.AS.bg.input, synapse=synapse),
													  'out': nengo.Probe(AS_net.AS.thalamus.output, synapse=synapse)}
								for AS_net in [net.routing_net, net.access_net] + net.broadcast_nets}})

			# Attentional levels
			self.probes.update({'attention': {p.label+" attention": nengo.Probe(p.attention.output, synapse=synapse)
								for p in self.senders}})

			# Receive levels
			self.probes.update({'receive': {p.label+" receive": nengo.Probe(p.receive.output, synapse=synapse)
								for p in self.receivers}})

	def run(self, T, simulator_cls, dt=.001):
		print("Number of neurons:", self.network.n_neurons)
		print("T:",T)

		self.send_seed()
		self.set_probes()
		with simulator_cls(self.network, dt=dt, seed=self.seed) as self.sim:
			self.sim.run(T)

