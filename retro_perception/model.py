import nengo
import nengo_spa as spa
import numpy as np
import random

class Delay(object):
	def __init__(self, dimensions, timesteps):
		self.history = np.zeros((timesteps, dimensions))
	def step(self, t, x):
		self.history = np.roll(self.history, -1, 0)
		self.history[-1] = x
		return self.history[0]


inhibition = .1
disinhibition = -1

class Processor(spa.Network):

	def __init__(
		self, 
		input_vocab, 
		output_vocab, 
		label,
		AM_mapping,
		feedback=.8, 
		AM_thr=0, 
		AM_cls=spa.WTAAssocMem, 
		#AM_fn=lambda x:x,
		npd_AM=50, 
		sender=True,
		receiver=True,
		seed=None, 
		add_to_container=None,
		add_ON=True
		):

		super(Processor, self).__init__(label, seed, add_to_container, vocabs=None)
		

		if isinstance(AM_mapping,list):
			AM_mapping = {k:k for k in AM_mapping}

		self.label = label
		self.sender = sender
		self.receiver = receiver
		
		# Domain specific processing
		add_ON = "+SOURCE*"+self.label if add_ON else ""
		self.AM = AM_cls(
			threshold=AM_thr, 
			input_vocab=input_vocab, 
			output_vocab=output_vocab,
			mapping={k:v+add_ON for k,v in AM_mapping.items()},
			#function=AM_fn,
			label=self.label+' AM',
			n_neurons = npd_AM
		)

		self.inhibs = {}
		for k in AM_mapping.keys():
			self.inhibs[k] = spa.Scalar()
			inhibition >> self.inhibs[k]

		for i,ens in enumerate(self.AM.all_ensembles):
			nengo.Connection(ens, ens, transform=feedback, synapse=.05)
			nengo.Connection(self.inhibs[list(AM_mapping.keys())[i]].output, ens, transform=-1)
		
		self.input = self.AM.input
		

		if self.sender	:
			
			if True: # Adds some proper delay to avoid operation recursion
				delay = .1 # in seconds
				self.preconscious = spa.Transcode(input_vocab=output_vocab, output_vocab=output_vocab)
				delayNode = nengo.Node(Delay(output_vocab.dimensions, int((delay) / .001)).step, 
					size_in=output_vocab.dimensions, 
					size_out=output_vocab.dimensions,
					label='delay node')

				pre_to_delay = nengo.Connection(self.AM.output, delayNode, 
					transform=np.ones((output_vocab.dimensions)),
					synapse=None)

				delay_to_post = nengo.Connection(delayNode, self.preconscious.input, synapse=None)
				self.preconscious = self.preconscious.output
			else:
				self.preconscious = self.AM.output


			self.attention = spa.Scalar()
			.5 >> self.attention

		if self.receiver :
			self.receive = spa.Scalar()

class Experiment():

	def __init__(self):
		self.last_trial = -9999
		self.stimuli = []
		self.cues = []

	def create_function(self, location, SOA):
		def function(t):
			if t-self.last_trial > 1:
				self.change(t)

			# return self.stimuli[location]+"+"+self.cues[location]
			if t-self.last_trial < SOA:
				return self.stimuli[location]
			elif t-self.last_trial < SOA+.3:
				return self.cues[location]
			else:
				return "0"
		return function

	def change(self, t):

		self.last_trial = t

		orientation = np.random.choice(["ORIENT_1","ORIENT_2"])
		self.stimuli = ["0","0"]
		location = np.random.choice([0,1])
		self.stimuli[location] = orientation

		self.cues = ["0","0"]
		self.cues[np.random.choice([0,1])] = ".75*CUE"


class Model():
	def __init__(self, vocab, SOA, proc_feedback, GW_feedback, GW_scale, BG_thr, BG_bias, n_neurons_per_dim, seed=None):
		self.vocab = vocab
		self.SOA = SOA
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

			self.experiment = Experiment()

			net.input_net = spa.Network(label='inputs', seed=self.seed)

			# We start defining the buffer slots in which information can
			# be placed:
			
			# A slot for the visual inputs
			

			V_mapping = {
				stim:'CONTENT*'+stim for stim in ['CUE*R','CUE*L','ORIENT_1*R','ORIENT_2*R','ORIENT_1*L','ORIENT_2*L']}
			print("keys",V_mapping.keys())
			with net.input_net:
				net.input_net.V_L_trans = spa.Transcode(function=self.experiment.create_function(0, self.SOA), output_vocab=self.vocab)
				net.input_net.V_R_trans = spa.Transcode(function=self.experiment.create_function(1, self.SOA), output_vocab=self.vocab)

				net.input_net.V_L = spa.State(self.vocab)
				net.input_net.V_R = spa.State(self.vocab)

				nengo.Connection(net.input_net.V_L_trans.output, net.input_net.V_L.input, synapse=None, transform=.1)
				nengo.Connection(net.input_net.V_R_trans.output, net.input_net.V_R.input, synapse=None, transform=.1)

				for V_loc in [net.input_net.V_L, net.input_net.V_R]:
					for ens in V_loc.all_ensembles:
						nengo.Connection(ens, ens, transform=.85, synapse=.01)

			net.V = Processor(
				self.vocab, self.vocab, 'V', 
				V_mapping, 
				feedback=self.proc_feedback,
				receiver=False, # V only sends info to GW
				npd_AM=self.n_neurons_per_dim['AM'], seed=self.seed)
			net.input_net.V_L * s.L >> net.V
			net.input_net.V_R * s.R >> net.V
			
			# A slot for the action (MORE or LESS)
			net.M = Processor(
				self.vocab, self.vocab, 'M',
				{stim:stim+'*CONTENT' for stim in ['ORIENT_1*R','ORIENT_2*R','ORIENT_1*L','ORIENT_2*L']},
				feedback=self.proc_feedback,
				sender=False, # M only receives info from GW
				npd_AM=self.n_neurons_per_dim['AM'], seed=self.seed)
					
			self.processors = [net.V, net.M]

			# Selects information from the processors
			net.GW_content = spa.WTAAssocMem(
				threshold=0, 
				input_vocab=self.vocab, 
				output_vocab=self.vocab,
				mapping=['CUE*R','CUE*L','ORIENT_1*R','ORIENT_2*R','ORIENT_1*L','ORIENT_2*L'],
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


					net.routing_net.labels.append("CUE_R")
					spa.ifmax(net.routing_net.labels[-1], self.BG_bias + spa.dot(net.GW_content, s.CUE*s.R),
									# disinhibition >> net.V.inhibs["ORIENT_1*R"],
									# disinhibition >> net.V.inhibs["ORIENT_2*R"],
									1 >> net.V.inhibs["ORIENT_1*L"],
									1 >> net.V.inhibs["ORIENT_2*L"],

									1 >> net.V.inhibs["CUE*R"],
									1 >> net.V.inhibs["CUE*L"],
								    *(.5 >> p.attention if p==net.V else -.5 >> p.attention for p in self.senders)
							 )

					net.routing_net.labels.append("CUE_L")
					spa.ifmax(net.routing_net.labels[-1], self.BG_bias + spa.dot(net.GW_content, s.CUE*s.L),
									# disinhibition >> net.V.inhibs["ORIENT_1*L"],
									# disinhibition >> net.V.inhibs["ORIENT_2*L"],
									1 >> net.V.inhibs["ORIENT_1*R"],
									1 >> net.V.inhibs["ORIENT_2*R"],

									1 >> net.V.inhibs["CUE*R"],
									1 >> net.V.inhibs["CUE*L"],
								    *(.5 >> p.attention if p==net.V else -.5 >> p.attention for p in self.senders)

							 )

					net.routing_net.labels.append("V_M")
					spa.ifmax(net.routing_net.labels[-1], self.BG_bias + self.BG_thr,
								  	1 >> net.M.receive,
								    *(.5 >> p.attention if p==net.V else -.5 >> p.attention for p in self.senders)

							 )            

					# net.routing_net.labels.append("Thresholder")
					# spa.ifmax(net.routing_net.labels[-1], self.BG_bias + self.BG_thr) # Threshold for action




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
			

			self.probes.update({'L': nengo.Probe(net.input_net.V_L.output, synapse=synapse)})
			self.probes.update({'R': nengo.Probe(net.input_net.V_R.output, synapse=synapse)})
			self.probes.update({'L_trans': nengo.Probe(net.input_net.V_L_trans.output, synapse=synapse)})
			self.probes.update({'R_trans': nengo.Probe(net.input_net.V_R_trans.output, synapse=synapse)})

			# GW
			self.probes.update({'GW': {GW.label: {  'in': nengo.Probe(GW.input, synapse=synapse),
													'out': nengo.Probe(GW.output, synapse=synapse)}
						for GW in [net.GW_source, net.GW_content]}})

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

