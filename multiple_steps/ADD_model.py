import nengo
import nengo_spa as spa
import numpy as np
from modules import AMProcessor, Button
import random
import pandas as pd
import matplotlib.pyplot as plt

import pytry

def create_vocab(D, seed, n_digits):
	pointers = ['D'+str(i) for i in range(n_digits+1)]
	pointers += [
		'FIXATE', #'MASK',
		'MORE','LESS',
		'GET','SET',
		'SIMPLE', 'CHAINED_ADD', 'CHAINED_SUB',
		'ON',
		'V', 'M'
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
		self.param('number of addition processors', n_ADD_proc=1)
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
		model = Model(p.vocab, p.n_ADD_proc, p.proc_feedback, p.proc_feedback_synapse, p.GW_feedback, p.GW_threshold, p.GW_scale, p.BG_thr, p.BG_bias, n_neurons_per_dim, p.seed, p.plt)
		model.make_probes()
		self.probes = model.probes
		return model.network

	def evaluate_behaviour(self, sim, p, seed):

		# get model's action
		model_behaviour = sim.data[self.probes['BTN']]
		print(model_behaviour)
		if np.count_nonzero(model_behaviour) > 1:
			acc = 0
		else:
			model_action = model_behaviour.sum()
			acc = int(model_action == p.n_ADD_proc+1)

		return dict(acc=acc)

	@staticmethod
	def plot_similarities(t_range, data, vocab, keys=False, autoscale=False, title='Similarity', sort_legend=True, subplot_nrows=0, subplot_ncols=0, subplot_i = 1):

		if not keys:
			keys = list(vocab.keys())

		if subplot_nrows * subplot_ncols > 0:
			plt.subplot(subplot_nrows,subplot_ncols,subplot_i)

		vectors = np.array([vocab.parse(p).v for p in keys])
		mean_activation = spa.similarity(data, vectors).mean(axis=0)
		sort_idx = np.argsort(mean_activation)[::-1]    

		ymin, ymax = -.25, 2.5
		plt.ylim(ymin, ymax)
		plt.autoscale(autoscale, axis='y')
		plt.grid(False)
		plt.plot(t_range, spa.similarity(data, vectors), linewidth=2.5)
		plt.title(title, size=15)
		if subplot_i==11:
			plt.xlabel("Time (s)", size=20, labelpad=20)
		# plt.ylabel("Similarity", size=15)
		# plt.xlim(left=t_range[0], right=t_range[-1])
		leg = plt.legend(keys, loc='upper center', ncol=5 if title=="Workspace source" else 4)
		# leg = plt.legend([k.replace('*CONTENT','')+': '+str(round(mean_activation[sort_idx][i],2)) for i,k in enumerate(np.array(keys)[sort_idx])], loc='upper center', ncol=3)
		
		# set the linewidth of each legend object
		for legobj in leg.legendHandles:
			legobj.set_linewidth(4.0)
			
		if subplot_nrows * subplot_ncols == 0:
			plt.show()
			
		# plt.yticks(range(3), range(3), fontsize=12 if (subplot_i-1)%3==0 else 0)
		# plt.xticks(range(1,7), range(1,7), fontsize=12 if subplot_i in [10,11,12] else 0)


		return subplot_i + 1

	def plot(self, sim, p):
		trange = sim.trange()
		ExperimentRun.plot_similarities(trange, sim.data[self.probes['GW']['out']], p.vocab, keys=['D'+str(p_i) for p_i in range(p.n_ADD_proc+1)] + ['FIXATE'], title='Workspace content')
		plt.show()
		ExperimentRun.plot_similarities(trange, sim.data[self.probes['processors']['M']['out']], p.vocab, keys=['D'+str(p_i) for p_i in range(p.n_ADD_proc+1)] + ['FIXATE'], title='M')
		plt.show()

		plt.plot(sim.trange(), sim.data[self.probes['BTN']])
		plt.show()

	def evaluate(self, p, sim, plt):
		try:
			sim.run(.5*(p.n_ADD_proc+2))
		except AssertionError:
			pass
		data = self.evaluate_behaviour(sim, p, p.seed)

		if plt:
			self.plot(sim, p)
		
		return data

class Model():
	def __init__(self, vocab, n_ADD_proc, proc_feedback, proc_feedback_synapse,
			GW_feedback, GW_threshold, GW_scale, BG_thr, BG_bias, n_neurons_per_dim, seed=None, plot=False):
		self.vocab = vocab
		self.n_ADD_proc = n_ADD_proc
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

			def V_input_function(t):
				if t<.5:
					return 'FIXATE'
				elif t<.6:
					return 'D0'
				else:
					return '0'

			V_mapping = ['D'+str(p_i) for p_i in range(self.n_ADD_proc+1)] + ['FIXATE']
			with net.input_net:
				net.input_net.RETINA_input = spa.Transcode(V_input_function, output_vocab=self.vocab)

			net.V = AMProcessor(
				self.vocab, self.vocab, 'V', 
				V_mapping, 
				feedback=self.proc_feedback,
				feedback_synapse=self.proc_feedback_synapse,
				receiver=False, # V only sends info to GW
				npd_AM=self.n_neurons_per_dim['AM'], seed=self.seed)
			nengo.Connection(net.input_net.RETINA_input.output, net.V.input.input, synapse=None)

			# A slot for the action (MORE or LESS)
			M_mapping = V_mapping
			net.M = AMProcessor(
				self.vocab, self.vocab, 'M',
				M_mapping,
				feedback=0,#self.proc_feedback,
				feedback_synapse=0,#self.proc_feedback_synapse,
				sender=False, # M only receives info from GW
				npd_AM=self.n_neurons_per_dim['AM'], seed=self.seed)
					
			net.BTN = nengo.Node(
				Button(SP_vectors=[self.vocab.parse(SP).v for SP in M_mapping]),
				size_in=self.vocab.dimensions)
			nengo.Connection(net.M.preconscious.output, net.BTN, synapse=None) # Connect output of M to BTN that records behavioral response

			
			self.ADD_dict = {
				'D'+str(p_i): AMProcessor(
					self.vocab, self.vocab, 'D'+str(p_i),
					{'D'+str(p_i): 'D'+str(p_i+1)},
					feedback=self.proc_feedback,
					feedback_synapse=self.proc_feedback_synapse,
					npd_AM=self.n_neurons_per_dim['AM'], seed=self.seed
				)
				for p_i in range(self.n_ADD_proc)
			}
			
			
			self.processors = [net.V, net.M] + list(self.ADD_dict.values())
			net.PREV = spa.State(self.vocab, feedback=1, feedback_synapse=.01, label='PREV')
			   
			# Selects information from the processors
			GW_mapping = V_mapping
			net.GW = spa.WTAAssocMem(
				threshold=self.GW_threshold, 
				input_vocab=self.vocab, 
				output_vocab=self.vocab,
				mapping=GW_mapping,
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

					for p_i in range(self.n_ADD_proc):
						set_condition = s.GET*self.vocab.parse('D'+str(p_i-1)) if p_i>0 else s.GET*self.vocab.parse('V')
						net.routing_net.labels.append("SET P"+str(p_i))
						spa.ifmax(net.routing_net.labels[-1], self.BG_bias
						+	spa.dot(net.PREV, set_condition)
						- 	spa.dot(net.GW, s.FIXATE) 	# too soon
						,
								  1 >> self.ADD_dict['D'+str(p_i)].receive,
								  .25*s.SET*self.vocab.parse('D'+str(p_i)) >> net.PREV,
							 )

						net.routing_net.labels.append("GET P"+str(p_i))
						spa.ifmax(net.routing_net.labels[-1], self.BG_bias
						+ 	spa.dot(net.PREV, s.SET*self.vocab.parse('D'+str(p_i))) * spa.dot(self.ADD_dict['D'+str(p_i)].preconscious, s.ON),
								  self.GW_scale >> self.ADD_dict['D'+str(p_i)].attention,
								  .25*s.GET*self.vocab.parse('D'+str(p_i)) >> net.PREV,
							 )

					net.routing_net.labels.append("SET M")
					spa.ifmax(net.routing_net.labels[-1], self.BG_bias 
					+ 	spa.dot(net.PREV, s.GET*self.vocab.parse('D'+str(self.n_ADD_proc-1)))
					+ 	.5*spa.dot(net.PREV, s.SET*s.M) # sustain
					,
								  1 >> net.M.receive,
								  .25*s.SET*s.M >> net.PREV,
							 )

					net.routing_net.labels.append("Thresholder")
					spa.ifmax(net.routing_net.labels[-1], self.BG_bias + self.BG_thr) # Threshold for action


		print(net.n_neurons)

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
				
				# GW
				self.probes.update({'GW': { 'in': nengo.Probe(net.GW.input, synapse=synapse),
											'out': nengo.Probe(net.GW.output, synapse=synapse)}
							})
				
				# PREV
				self.probes.update({net.PREV.label: nengo.Probe(net.PREV.input, synapse=synapse)})
						
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


