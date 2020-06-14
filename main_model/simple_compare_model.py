import nengo
import nengo_spa as spa
import numpy as np
from modules import Processor, Button, DirectCompareProcessor
import random

import matplotlib.pyplot as pyplot


import pytry

class ExperimentRun(pytry.NengoTrial):
	def params(self):
		self.param('scaling of the number of neurons', n_neurons_scale=1)
		self.param('crosstalk strength', s_crosstalk=0)
		self.param('integrator evidence strength', s_evidence=0)
		self.param('vocab', vocab=None)
		self.param('experiment', xp=None)
		self.param('feedback of processors', proc_feedback=0)
		self.param('feedback of global workspace', GW_feedback=0)
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
		model = Model(p.vocab, p.xp, p.s_crosstalk, p.s_evidence, p.proc_feedback, p.GW_feedback, p.GW_scale, p.BG_thr, p.BG_bias, n_neurons_per_dim, p.seed)
		model.make_probes()
		self.probes = model.probes
		print("Number of neurons:", model.network.n_neurons)
		return model.network

	def evaluate_behaviour(self, sim, p):

		feedbacks = [] # 0:no answer / 1:wrong answer / 2:correct answer
		model_actions = []
		RTs = []

		t = 0
		while t<p.xp.T-.01:
			t += p.xp.trial_length
			trial = p.xp(t)[0]
			expected_action = trial.expected_action
			t_window = (np.where(np.logical_and(sim.trange() < t, sim.trange() > t-p.xp.trial_length))[0],)
			
			# get model's action
			model_behaviour = sim.data[self.probes['BTN']][t_window]
			if np.count_nonzero(model_behaviour) > 1:
				raise ValueError("more than one action")
			
			model_action = model_behaviour.sum()
			model_actions.append(int(model_action))
			if model_action == 0:
				RTs.append(p.xp.trial_length) # (arbitrary?)
				feedbacks.append(0)
			else:
				action_t_idx = np.nonzero(model_behaviour[:,0])
				RTs.append(sim.trange()[t_window][action_t_idx][0] - (t-p.xp.trial_length) - p.xp.t_start)
				feedbacks.append(int(model_action==expected_action) + 1)
			
		return dict(feedbacks=feedbacks, RTs=RTs, model_actions=model_actions)

	def evaluate(self, p, sim, plt):
		sim.run(p.xp.T)
		result = self.evaluate_behaviour(sim,p)

		# SPs = ['D2', 'D4', 'D6', 'D8']
		# pyplot.plot(sim.trange(), spa.similarity(sim.data[self.probes['processors']['COM']['in']], [p.vocab.parse(SP) for SP in SPs]))
		# pyplot.title("Input similarity")
		# pyplot.xlabel("Time")
		# pyplot.legend(SPs)
		# pyplot.show()

		# SPs = ['MORE*CONTENT', 'LESS*CONTENT']
		# pyplot.plot(sim.trange(), spa.similarity(sim.data[self.probes['processors']['COM']['out']], [p.vocab.parse(SP) for SP in SPs]))
		# pyplot.title("Output similarity")
		# pyplot.xlabel("Time")
		# pyplot.legend(SPs)
		# pyplot.show()

		# # pyplot.plot(sim.trange(), sim.data[self.probes['processors']['COM']['compare']])
		# # pyplot.show()

		# pyplot.plot(sim.trange(), sim.data[self.probes['processors']['COM']['controlled']])
		# pyplot.show()

		# pyplot.plot(sim.trange(), sim.data[self.probes['processors']['COM']['accumulator']])
		# pyplot.show()

		return result

class Model():
	def __init__(self, vocab, experiment, s_crosstalk, s_evidence, proc_feedback, GW_feedback, GW_scale, BG_thr, BG_bias, n_neurons_per_dim, seed=None):
		self.vocab = vocab
		self.experiment = experiment
		self.s_crosstalk = s_crosstalk
		self.s_evidence = s_evidence
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
			
			# # A slot for the goal/task
			# net.G = spa.State(self.vocab, label='G')
			# with net.input_net:
			# 	net.input_net.G_input = spa.Transcode(self.experiment.G_input, output_vocab=self.vocab)
			# net.input_net.G_input >> net.G
			
			# A slot for the visual input (the digit N). Feedback is used for iconic memory (100-300ms)
			# V_mapping = {
			# 	digit:'CONTENT*'+digit for digit in ['D2','D4','D6','D8','FIXATE','MASK']}
			with net.input_net:
				net.input_net.RETINA_input = spa.Transcode(self.experiment.RETINA_input, output_vocab=self.vocab)
				
			
			# # A slot for the action (MORE or LESS)
			# net.M = Processor(
			# 	self.vocab, self.vocab, 'M',
			# 	{stim:stim+'*CONTENT' for stim in ['MORE','LESS']},
			# 	feedback=self.proc_feedback,
			# 	sender=False, # M only receives info from GW
			# 	npd_AM=self.n_neurons_per_dim['AM'], seed=self.seed)
					
			

			# # An associative memory for the + operation
			# net.ADD = Processor(
			# 	self.vocab, self.vocab, 'ADD',
			# 	{   'D2':'D4*CONTENT',
			# 		'D4':'D6*CONTENT',
			# 		'D6':'D8*CONTENT',
			# 		'D8':'D2*CONTENT'   },
			# 	feedback=self.proc_feedback,
			# 	npd_AM=self.n_neurons_per_dim['AM'], seed=self.seed
			# )
			
			# # An associative memory for the - operation
			# net.SUB = Processor(
			# 	self.vocab, self.vocab, 'SUB',
			# 	{   'D2':'D8*CONTENT',
			# 		'D4':'D2*CONTENT',
			# 		'D6':'D4*CONTENT',
			# 		'D8':'D6*CONTENT'   },
			# 	feedback=self.proc_feedback,
			# 	npd_AM=self.n_neurons_per_dim['AM'], seed=self.seed
			# )
			
			# An associative memory for the "compare to 5" operation
			if self.s_evidence:
				net.COM = DirectCompareProcessor(
					self.vocab, 
					self.vocab, 
					'COM',
					self.s_evidence,
					self.proc_feedback, 
					self.n_neurons_per_dim
					)
			else:
				net.COM = Processor(
					self.vocab, self.vocab, 'COM',
					{   'D2':'LESS*CONTENT',
						'D4':'LESS*CONTENT',
						'D6':'MORE*CONTENT',
						'D8':'MORE*CONTENT'  },
					feedback=self.proc_feedback,
					npd_AM=self.n_neurons_per_dim['AM'], seed=self.seed
				)
			# if self.s_crosstalk:
			# 	nengo.Connection(net.V.preconscious, net.COM.input, transform=self.s_crosstalk)

			nengo.Connection(net.input_net.RETINA_input.output, net.COM.input, synapse=None)

			net.BTN = nengo.Node(Button(
				[self.vocab.parse('CONTENT*LESS').v, self.vocab.parse('CONTENT*MORE').v], 
				self.experiment.trial_length,
				wait_length=.5), 
				size_in=self.vocab.dimensions)
			nengo.Connection(net.COM.output.output, net.BTN) # Connect output of M to BTN that records behavioral response
			
			
			# self.processors = [net.V, net.ADD, net.SUB, net.COM, net.M]
			self.processors = [net.COM]

			# # Selects information from the processors
			# net.GW_content = spa.WTAAssocMem(
			# 	threshold=0, 
			# 	input_vocab=self.vocab, 
			# 	output_vocab=self.vocab,
			# 	mapping=['D2','D4','D6','D8', 'FIXATE','MASK', 'MORE','LESS'],
			# 	function=lambda x:x>0,
			# 	label='GW content',
			# 	n_neurons = self.n_neurons_per_dim['AM']
			# )
			# for ens in net.GW_content.all_ensembles: # Add feedback to each ensemble
			# 	nengo.Connection(ens, ens, transform=self.GW_feedback, synapse=.05)
			
			# # Selects source of information (i.e. determines which processor sent the information)
			# net.GW_source = spa.WTAAssocMem(
			# 	threshold=0, 
			# 	input_vocab=self.vocab, 
			# 	output_vocab=self.vocab,
			# 	mapping=[p.label for p in self.senders],
			# 	function=lambda x:x>0,
			# 	label='GW source',
			# 	n_neurons = self.n_neurons_per_dim['AM']
			# )
			# for ens in net.GW_source.all_ensembles: # Add feedback to each ensemble
			# 	nengo.Connection(ens, ens, transform=self.GW_feedback, synapse=.05)
			
			# # net.PREV = spa.ThresholdingAssocMem(
			# # 	threshold=0, 
			# # 	input_vocab=self.vocab, 
			# # 	output_vocab=self.vocab,
			# # 	mapping=['FIXATE','V','COM','ADD','SUB'],
			# # 	label='PREV',
			# # 	n_neurons = self.n_neurons_per_dim['AM']
			# # )
			# # for ens in net.PREV.all_ensembles: # Add feedback to each ensemble
			# # 	nengo.Connection(ens, ens, transform=.75, synapse=.05)
			# net.PREV_inp = spa.Transcode(input_vocab=self.vocab, output_vocab=self.vocab, label='PREV input')
			# net.PREV_out = spa.State(self.vocab, feedback=.75, label='PREV')
			# nengo.Connection(net.PREV_inp.output, net.PREV_out.input, synapse=.15) # beware of implausible values!

			
			
			# # access network
			# with spa.Network(label='access', seed=self.seed) as net.access_net:
			# 	net.access_net.labels = []
			# 	with spa.ActionSelection() as net.access_net.AS:
			# 		for p in self.senders:
			# 			net.access_net.labels.append(p.label)
			# 			spa.ifmax(p.label, self.BG_bias+spa.dot(p.preconscious, s.SOURCE*self.vocab.parse(p.label)) * p.attention,

			# 						  # 100 * self.GW_scale * p.preconscious * ~s.CONTENT >> net.GW_content.input,
			# 						  self.GW_scale * p.preconscious * ~s.CONTENT >> net.GW_content.input,
			# 						  self.GW_scale * self.vocab.parse(p.label) >> net.GW_source.input
			# 						 )
			# 		net.access_net.labels.append("Thresholder")
			# 		spa.ifmax(self.BG_bias + self.BG_thr) 
			
			# # broadcast networks
			# net.broadcast_nets = []
			# for p in self.receivers: # each processor p receives GW's content if p's "receive" level is more than a threshold
			# 	net.broadcast_nets.append(spa.Network(label='broadcast '+p.label, seed=self.seed))
			# 	net.broadcast_nets[-1].labels = []
			# 	with net.broadcast_nets[-1]:
			# 		with spa.ActionSelection() as net.broadcast_nets[-1].AS:
			# 			net.broadcast_nets[-1].labels.append(p.label+" GO")
			# 			spa.ifmax(net.broadcast_nets[-1].labels[-1], p.receive,
			# 						 net.GW_content >> p.input
			# 					 )
			# 			net.broadcast_nets[-1].labels.append(p.label+" NOGO")
			# 			spa.ifmax(net.broadcast_nets[-1].labels[-1], .5)
			   
			# # routing network
			# with spa.Network(label='routing', seed=self.seed) as net.routing_net:
			# 	net.routing_net.labels = []
			# 	with spa.ActionSelection() as net.routing_net.AS:
					
			# 		net.routing_net.labels.append("GET V")
			# 		spa.ifmax(net.routing_net.labels[-1],  self.BG_bias + spa.dot(net.GW_source, s.V) * spa.dot(net.GW_content, s.FIXATE),
			# 					  *(.5 >> p.attention if p==net.V else -.5 >> p.attention for p in self.senders),
			# 					  s.GET*s.V >> net.PREV_inp
			# 				 )
					
			# 		net.routing_net.labels.append("SET COM")
			# 		spa.ifmax(net.routing_net.labels[-1], self.BG_bias
			# 		+	spa.dot(net.GW_source, s.V) * spa.dot(net.G, s.SIMPLE)
			# 		+	spa.dot(net.GW_source, s.SUB) * spa.dot(net.G, s.CHAINED_SUB)
			# 		+	spa.dot(net.GW_source, s.ADD) * spa.dot(net.G, s.CHAINED_ADD)
			# 		- 	spa.dot(net.GW_content, s.FIXATE) 	# too soon
			# 		- 	spa.dot(net.GW_content, s.MASK) 	# too late
			# 		- 	spa.dot(net.PREV_out, s.SET*s.COM),  	# already done
			# 					  1 >> net.COM.receive,
			# 					  -.5 >> net.COM.attention, # block access
			# 					  s.SET*s.COM >> net.PREV_inp,
			# 				 )

			# 		net.routing_net.labels.append("GET COM")
			# 		spa.ifmax(net.routing_net.labels[-1], self.BG_bias
			# 		+ 	(spa.dot(net.PREV_out, s.SET*s.COM) * spa.dot(net.COM.preconscious, s.SOURCE*self.vocab.parse("COM_SOURCE")))
			# 		- 	spa.dot(net.PREV_out, s.GET*s.COM),		# already done
			# 					  *(.5 >> p.attention if p==net.COM else -.5 >> p.attention for p in self.senders),
			# 					  s.GET*s.COM >> net.PREV_inp
			# 				 )

			# 		net.routing_net.labels.append("SET SUB")
			# 		spa.ifmax(net.routing_net.labels[-1], self.BG_bias 
			# 		+ 	spa.dot(net.GW_source, s.V) * spa.dot(net.G, s.CHAINED_SUB)
			# 		- 	spa.dot(net.GW_content, s.FIXATE) 	# too soon
			# 		- 	spa.dot(net.GW_content, s.MASK) 	# too late
			# 		- 	spa.dot(net.PREV_out, s.SET*s.SUB),		# already done
			# 					  1 >> net.SUB.receive,
			# 					  -.5 >> net.SUB.attention, # block access
			# 					  s.SET*s.SUB >> net.PREV_inp,
			# 				 )

			# 		net.routing_net.labels.append("GET SUB")
			# 		spa.ifmax(net.routing_net.labels[-1], self.BG_bias
			# 		+ 	(spa.dot(net.PREV_out, s.SET*s.SUB) * spa.dot(net.SUB.preconscious, s.SOURCE*self.vocab.parse("SUB_SOURCE")))
			# 		- 	spa.dot(net.PREV_out, s.GET*s.SUB),		# already done,
			# 					  *(.5 >> p.attention if p==net.SUB else -.5 >> p.attention for p in self.senders),
			# 					  s.GET*s.SUB >> net.PREV_inp,
			# 				 )

			# 		net.routing_net.labels.append("SET ADD")
			# 		spa.ifmax(net.routing_net.labels[-1], self.BG_bias 
			# 		+ 	spa.dot(net.GW_source, s.V) * spa.dot(net.G, s.CHAINED_ADD)
			# 		- 	spa.dot(net.GW_content, s.FIXATE) 	# too soon
			# 		- 	spa.dot(net.GW_content, s.MASK) 	# too late
			# 		- 	spa.dot(net.PREV_out, s.SET*s.ADD),		# already done
			# 					  1 >> net.ADD.receive,
			# 					  -.5 >> net.ADD.attention, # block access
			# 					  s.SET*s.ADD >> net.PREV_inp,
			# 				 )

			# 		net.routing_net.labels.append("GET ADD")
			# 		spa.ifmax(net.routing_net.labels[-1], self.BG_bias
			# 		+ 	(spa.dot(net.PREV_out, s.SET*s.ADD) * spa.dot(net.ADD.preconscious, s.SOURCE*self.vocab.parse("ADD_SOURCE")))
			# 		- 	spa.dot(net.PREV_out, s.GET*s.ADD),		# already done,
			# 					  *(.5 >> p.attention if p==net.ADD else -.5 >> p.attention for p in self.senders),
			# 					  s.GET*s.ADD >> net.PREV_inp,
			# 				 )

			# 		net.routing_net.labels.append("SET M")
			# 		spa.ifmax(net.routing_net.labels[-1], self.BG_bias 
			# 		+ 	spa.dot(net.GW_source, s.COM)
			# 		- 	spa.dot(net.PREV_out, s.SET*s.M),		# already done
			# 					  1 >> net.M.receive,
			# 					  s.SET*s.M >> net.PREV_inp,
			# 				 )

			# 		net.routing_net.labels.append("Thresholder")
			# 		spa.ifmax(net.routing_net.labels[-1], self.BG_bias + self.BG_thr) # Threshold for action




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

	def make_probes(self, synapse=.015):
		net = self.network
		with net:
			# Processors
			self.probes = {'processors': {p.label: {'in': nengo.Probe(p.input, synapse=synapse),
													# 'out': nengo.Probe(p.AM.output, synapse=synapse) if p != net.COM else nengo.Probe(p.preconscious, synapse=synapse)}
													'out': nengo.Probe(p.preconscious, synapse=synapse)}
								for p in self.processors}}
			
			self.probes.update({'BTN': nengo.Probe(net.BTN)})

			self.probes['processors']['COM'].update({
				# 'compare': nengo.Probe(net.COM.compare, synapse=synapse),
				'controlled': nengo.Probe(net.COM.controlled, synapse=synapse),
				'accumulator': nengo.Probe(net.COM.accumulator, synapse=synapse)
				})

			# # GW
			# self.probes.update({'GW': {GW.label: {  'in': nengo.Probe(GW.input, synapse=synapse),
			# 										'out': nengo.Probe(GW.output, synapse=synapse)}
			# 			for GW in [net.GW_source, net.GW_content]}})

			# # PREV and G
			# self.probes.update({state.label: nengo.Probe(state.input, synapse=synapse)
			# 			for state in [net.PREV_out, net.G]})
					
			# # Action selection networks
			# self.probes.update({'AS_nets': {AS_net.label: {'in': nengo.Probe(AS_net.AS.bg.input, synapse=synapse),
			# 										  'out': nengo.Probe(AS_net.AS.thalamus.output, synapse=synapse)}
			# 					for AS_net in [net.routing_net, net.access_net] + net.broadcast_nets}})

			# # Attentional levels
			# self.probes.update({'attention': {p.label+" attention": nengo.Probe(p.attention.output, synapse=synapse)
			# 					for p in self.senders}})

			# # Receive levels
			# self.probes.update({'receive': {p.label+" receive": nengo.Probe(p.receive.output, synapse=synapse)
			# 					for p in self.receivers}})

	def run(self, simulator_cls, dt=.001):
		print("Number of neurons:", self.network.n_neurons)
		
		# not sure if it's a good way, because many intermediate nodes in SPA modules
		n_synapses = 0
		for c in self.network.all_connections:
			size_pre = c.pre.n_neurons if isinstance(c.pre, nengo.Ensemble) else 1
			size_post = c.post.n_neurons if isinstance(c.post, nengo.Ensemble) else 1
			n_synapses += size_pre*size_post
		print("(probably wrong) Number of synapses:", n_synapses)
		print("T:", self.experiment.trial_length*len(self.experiment.trials))

		self.send_seed()
		self.make_probes()
		with simulator_cls(self.network, dt=dt, seed=self.seed) as sim:
			sim.run(self.experiment.trial_length*len(self.experiment.trials))

		return sim

