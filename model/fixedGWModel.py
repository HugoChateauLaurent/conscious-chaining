import nengo
import nengo_spa as spa
import numpy as np
from modules import AMProcessor, Button, Processor, DirectCompareProcessor, Delay
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
		self.param('scaling of the number of neurons in combined ensemble', n_neurons_scale_combined=1)
		self.param('crosstalk strength', s_crosstalk=0)
		self.param('integrator evidence strength', s_evidence=0)
		self.param('number of samples for comparison function', n_samples=1000)
		self.param('duration of unmodelled visual processing', t_senso=0)
		self.param('vocab', vocab=None)
		self.param('experiment', xp=None)
		self.param('feedback of processors', proc_feedback=0)
		self.param('feedback synapse of processors', proc_feedback_synapse=.1)
		self.param('feedback synapse of previour routing', prev_feedback_synapse=.1)
		self.param('feedback of global workspace', GW_feedback=0)
		self.param('threshold for global workspace WTA networks', GW_threshold=.5)
		self.param('amplification of global workspace input signal', GW_scale=20)
		self.param('threshold for action selection networks', BG_thr=.1)
		self.param('bias for action selection networks', BG_bias=.5)
		self.param('whether to reset the comparison integrator at the beginning of each trial', integrator_reset=True)

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
		n_neurons_per_dim['combined'] = int(50*p.n_neurons_scale_combined)

		return n_neurons_per_dim

	def model(self, p):
		n_neurons_per_dim = self.make_n_neurons_per_dim(p)
		model = Model(p.vocab, p.xp, p.s_crosstalk, p.s_evidence, p.n_samples, p.integrator_reset, p.t_senso, p.proc_feedback, p.proc_feedback_synapse, p.prev_feedback_synapse, p.GW_feedback, p.GW_threshold, p.GW_scale, p.BG_thr, p.BG_bias, n_neurons_per_dim, p.seed, p.plt)
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
		subplot_nrows=4
		subplot_ncols=3
		plt.figure(figsize=(6*subplot_ncols,2.5*subplot_nrows))
		
		trial_t = lambda trial_number: trial_number*p.xp.trial_length
		
		focus_start = 0 # first trial to plot
		n_focus = 200 # how many trials to plot
		start = trial_t(focus_start)
		end = trial_t(focus_start+n_focus)
		skip = 1
		trange = sim.trange()
		selected_idx = np.where(np.logical_and(trange > start, trange < end))
		trange = trange[selected_idx][::skip]


		subplot_i=ExperimentRun.plot_similarities(trange, sim.data[self.probes['G']][selected_idx][::skip], p.vocab, keys=['SIMPLE','CHAINED_ADD','CHAINED_SUB'], title='Task', subplot_nrows=subplot_nrows, subplot_ncols=subplot_ncols)
		subplot_i=ExperimentRun.plot_similarities(trange, sim.data[self.probes['GW']['out']][selected_idx][::skip], p.vocab, keys=['D2','D4','D6','D8','MORE','LESS','FIXATE'], title='Workspace content', subplot_i=subplot_i, subplot_nrows=subplot_nrows, subplot_ncols=subplot_ncols)
		subplot_i=ExperimentRun.plot_similarities(trange, sim.data[self.probes['processors']['V']['out']][selected_idx][::skip], p.vocab, keys=['D2','D4','D6','D8','FIXATE'], title='Visual', subplot_i=subplot_i, subplot_nrows=subplot_nrows, subplot_ncols=subplot_ncols)
		subplot_i=ExperimentRun.plot_similarities(trange, sim.data[self.probes['PREV']][selected_idx][::skip], p.vocab, keys=['GET*V','SET*COM','GET*COM','SET*ADD','GET*ADD','SET*SUB','GET*SUB','SET*M'], title='Previous routing', subplot_i=subplot_i, subplot_nrows=subplot_nrows, subplot_ncols=subplot_ncols)
		subplot_i+=1
		subplot_i=ExperimentRun.plot_similarities(trange, sim.data[self.probes['processors']['M']['out']][selected_idx][::skip], p.vocab, keys=['MORE','LESS'], title='Motor', subplot_i=subplot_i, subplot_nrows=subplot_nrows, subplot_ncols=subplot_ncols)
		subplot_i=ExperimentRun.plot_similarities(trange, sim.data[self.probes['processors']['ADD']['in']][selected_idx][::skip], p.vocab, keys=['D2','D4','D6','D8','FIXATE'], title='Add input', subplot_i=subplot_i, subplot_nrows=subplot_nrows, subplot_ncols=subplot_ncols)
		subplot_i=ExperimentRun.plot_similarities(trange, sim.data[self.probes['processors']['SUB']['in']][selected_idx][::skip], p.vocab, keys=['D2','D4','D6','D8','FIXATE'], title='Subtract input', subplot_i=subplot_i, subplot_nrows=subplot_nrows, subplot_ncols=subplot_ncols)
		subplot_i=ExperimentRun.plot_similarities(trange, sim.data[self.probes['processors']['COM']['in']][selected_idx][::skip], p.vocab, keys=['D2','D4','D6','D8','FIXATE'], title='Compare input', subplot_i=subplot_i, subplot_nrows=subplot_nrows, subplot_ncols=subplot_ncols)
		subplot_i=ExperimentRun.plot_similarities(trange, sim.data[self.probes['processors']['ADD']['out']][selected_idx][::skip], p.vocab, keys=['D2','D4','D6','D8'], title='Add output', subplot_i=subplot_i, subplot_nrows=subplot_nrows, subplot_ncols=subplot_ncols)
		subplot_i=ExperimentRun.plot_similarities(trange, sim.data[self.probes['processors']['SUB']['out']][selected_idx][::skip], p.vocab, keys=['D2','D4','D6','D8'], title='Subtract output', subplot_i=subplot_i, subplot_nrows=subplot_nrows, subplot_ncols=subplot_ncols)
		subplot_i=ExperimentRun.plot_similarities(trange, sim.data[self.probes['processors']['COM']['out']][selected_idx][::skip], p.vocab, keys=['MORE','LESS'], title='Compare output', subplot_i=subplot_i, subplot_nrows=subplot_nrows, subplot_ncols=subplot_ncols)

		plt.subplots_adjust(hspace=0.3, wspace=0.02)
		
		# plt.savefig("figures/model_beha.eps")

		plt.show()
		


	def evaluate(self, p, sim, plt):
		sim.run(p.xp.T)
		data = self.evaluate_behaviour(sim, p, p.seed)
		if plt:
			self.plot(sim, p)

		return data

class Model():
	def __init__(self, vocab, experiment, s_crosstalk, s_evidence, n_samples, integrator_reset, t_senso, proc_feedback, proc_feedback_synapse,
			prev_feedback_synapse, GW_feedback, GW_threshold, GW_scale, BG_thr, BG_bias, n_neurons_per_dim, seed=None, plot=False):
		self.vocab = vocab
		self.experiment = experiment
		self.s_crosstalk = s_crosstalk
		self.s_evidence = s_evidence
		self.n_samples = n_samples
		self.integrator_reset = integrator_reset
		self.t_senso = t_senso
		self.proc_feedback = proc_feedback
		self.proc_feedback_synapse = proc_feedback_synapse
		self.prev_feedback_synapse = prev_feedback_synapse
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
			
			# A slot for the goal/task
			net.G = spa.State(self.vocab, label='G')
			with net.input_net:
				net.input_net.G_input = spa.Transcode(self.experiment.G_input, output_vocab=self.vocab)
			net.input_net.G_input >> net.G
			
			# A slot for the visual input (the digit N). Feedback is used for iconic memory (100-300ms)
			V_mapping = ['D2','D4','D6','D8','FIXATE']#,'MASK']
			with net.input_net:
				net.input_net.RETINA_input = spa.Transcode(self.experiment.RETINA_input, output_vocab=self.vocab)
				net.input_net.senso_delay = nengo.Node(Delay(self.vocab.dimensions, self.t_senso).step, size_in=self.vocab.dimensions, size_out=self.vocab.dimensions) # sensory processing delay
				nengo.Connection(net.input_net.RETINA_input.output, net.input_net.senso_delay, synapse=None)

			net.V = Processor(
				self.vocab, self.vocab, 'V', 
				receiver=False, # V only sends info to GW
				seed=self.seed
			)
			nengo.Connection(net.input_net.senso_delay, net.V.input.input, synapse=None)
			net.V.state = spa.State(self.vocab, feedback=self.proc_feedback, feedback_synapse=self.proc_feedback_synapse)
			nengo.Connection(net.V.input.output, net.V.state.input, synapse=None)
			nengo.Connection(net.V.state.output, net.V.preconscious.input, synapse=None)

			
			# A slot for the action (MORE or LESS)
			net.M = Processor(
				self.vocab, self.vocab, 'M',
				sender=False, # M only receives info from GW
				seed=self.seed
			)
			nengo.Connection(net.M.input.output, net.M.preconscious.input, synapse=None)

					
			net.BTN = nengo.Node(Button(
				SP_vectors=[self.vocab.parse('LESS').v, self.vocab.parse('MORE').v], 
				trial_length=self.experiment.trial_length,
				wait_length=self.experiment.t_start), 
				size_in=self.vocab.dimensions)
			nengo.Connection(net.M.preconscious.output, net.BTN) # Connect output of M to BTN that records behavioral response

			# An associative memory for the + operation
			net.ADD = AMProcessor(
				self.vocab, self.vocab, 'ADD',
				{   'D2':'D4',
					'D4':'D6',
					'D6':'D8',
					'D8':'D2'   },
				feedback=self.proc_feedback,
				feedback_synapse=self.proc_feedback_synapse,
				npd_AM=self.n_neurons_per_dim['AM'], seed=self.seed
			)
			
			# An associative memory for the - operation
			net.SUB = AMProcessor(
				self.vocab, self.vocab, 'SUB',
				{   'D2':'D8',
					'D4':'D2',
					'D6':'D4',
					'D8':'D6'   },
				feedback=self.proc_feedback,
				feedback_synapse=self.proc_feedback_synapse,
				npd_AM=self.n_neurons_per_dim['AM'], seed=self.seed
			)
			
			# An associative memory for the "compare to 5" operation
			if self.s_evidence is not None:
				net.COM = DirectCompareProcessor(
					self.vocab, 
					self.vocab, 
					'COM',
					self.s_evidence,
					self.proc_feedback, 
					self.proc_feedback_synapse,
					self.experiment.trial_length if self.integrator_reset else None,
					self.n_neurons_per_dim,
					self.rng,
					self.n_samples
					)
			else:
				net.COM = AMProcessor(
					self.vocab, self.vocab, 'COM',
					{   'D2':'LESS',
						'D4':'LESS',
						'D6':'MORE',
						'D8':'MORE'  },
					feedback=self.proc_feedback,
					feedback_synapse=self.proc_feedback_synapse,
					npd_AM=self.n_neurons_per_dim['AM'], seed=self.seed
				)
			if self.s_crosstalk:
				nengo.Connection(net.V.preconscious.output, net.COM.input.input, transform=self.s_crosstalk)
			
			
			self.processors = [net.V, net.ADD, net.SUB, net.COM, net.M]

			# net.PREV = spa.Transcode(input_vocab=self.vocab, output_vocab=self.vocab, label='PREV input')
			net.PREV = spa.State(self.vocab, feedback=1, feedback_synapse=self.prev_feedback_synapse, label='PREV')
			# nengo.Connection(net.PREV.output, net.PREV.input, synapse=.005)
			   
			# Selects information from the processors
			net.GW = spa.WTAAssocMem(
				threshold=0,
				input_vocab=self.vocab, 
				output_vocab=self.vocab,
				mapping=['D2','D4','D6','D8', 'FIXATE', 'MORE','LESS'],#,'MASK'],
				function=lambda x:x>0,
				label='GW content',
				n_neurons = self.n_neurons_per_dim['AM']
			)

			# after thesis change (set WTA threshold to 0)
			nengo.Connection(net.GW.output, net.GW.input,  # feedback
				transform=self.GW_threshold, synapse=.02)#self.GW_feedback, synapse=.02)

			for p in self.senders:
				nengo.Connection(p.attention_weighting.output, net.GW.input, synapse=None)
				# p.sent = spa.Compare(self.vocab)
				# nengo.Connection(p.attention_weighting.output, p.sent.input_a, synapse=None)#, transform=1/self.GW_scale)
				# nengo.Connection(net.GW.output, p.sent.input_b, synapse=None)
				# nengo.Connection(p.sent.output, net.PREV.input, transform=[[v] for v in self.vocab.parse('GET*'+p.label).v])
			for p in self.receivers:
				nengo.Connection(net.GW.output, p.broadcast.output, synapse=None)
			# 	p.received = spa.Compare(self.vocab)
			# 	nengo.Connection(p.receive_weighting.output, p.received.input_a, synapse=None)
			# 	nengo.Connection(net.GW.output, p.received.input_b, synapse=None)
			# 	# nengo.Connection(p.received.output, net.PREV.input, transform=[[v] for v in self.vocab.parse('SET*'+p.label).v])

			# # PREV selection network
			# with spa.Network(label='PREV selection', seed=self.seed) as net.PREV_selection_net:
			# 	net.PREV_selection_net.labels = []
			# 	with spa.ActionSelection() as net.PREV_selection_net.AS:
			# 		for p in self.senders:
			# 			net.PREV_selection_net.labels.append("GET "+p.label)
			# 			spa.ifmax(net.PREV_selection_net.labels[-1],  self.BG_bias + p.sent,
			# 						  .25*s.GET*self.vocab.parse(p.label) >> net.PREV
			# 					 )

			# 		for p in self.receivers:
			# 			net.PREV_selection_net.labels.append("SET "+p.label)
			# 			spa.ifmax(net.PREV_selection_net.labels[-1],  self.BG_bias + p.received,
			# 						  .25*s.SET*self.vocab.parse(p.label) >> net.PREV
			# 					 )

			# 		net.PREV_selection_net.labels.append("Thresholder")
			# 		spa.ifmax(net.PREV_selection_net.labels[-1], self.BG_bias + self.BG_thr) # Threshold 

			# routing network
			with spa.Network(label='routing', seed=self.seed) as net.routing_net:
				net.routing_net.labels = []
				with spa.ActionSelection() as net.routing_net.AS:
					
					net.routing_net.labels.append("GET V")
					spa.ifmax(net.routing_net.labels[-1],  self.BG_bias + spa.dot(net.V.preconscious, s.FIXATE) + spa.dot(net.GW, s.FIXATE),
								  self.GW_scale >> net.V.attention,
								  .25*s.GET*s.V >> net.PREV
							 )
					
					net.routing_net.labels.append("SET COM")
					spa.ifmax(net.routing_net.labels[-1], self.BG_bias
					+	spa.dot(net.PREV, s.GET*s.V) * spa.dot(net.G, s.SIMPLE)
					+	spa.dot(net.PREV, s.GET*s.SUB) * spa.dot(net.G, s.CHAINED_SUB)
					+	spa.dot(net.PREV, s.GET*s.ADD) * spa.dot(net.G, s.CHAINED_ADD)
					- 	spa.dot(net.GW, s.FIXATE) 	# too soon
					- 	spa.dot(net.PREV, s.GET*s.COM) # already done
					,
								  1 >> net.COM.receive,
								  .25*s.SET*s.COM >> net.PREV,
							 )

					net.routing_net.labels.append("GET COM")
					spa.ifmax(net.routing_net.labels[-1], self.BG_bias
					+ 	spa.dot(net.PREV, s.SET*s.COM) * spa.dot(net.COM.preconscious, s.ON),
								  self.GW_scale >> net.COM.attention,
								  .25*s.GET*s.COM >> net.PREV
							 )

					net.routing_net.labels.append("SET SUB")
					spa.ifmax(net.routing_net.labels[-1], self.BG_bias 
					+ 	spa.dot(net.PREV, s.GET*s.V) * spa.dot(net.G, s.CHAINED_SUB)
					- 	spa.dot(net.GW, s.FIXATE) 	# too soon
					- 	spa.dot(net.PREV, s.GET*s.SUB) # already done
					,
								  1 >> net.SUB.receive,
								  .25*s.SET*s.SUB >> net.PREV,
							 )

					net.routing_net.labels.append("GET SUB")
					spa.ifmax(net.routing_net.labels[-1], self.BG_bias
					+ 	spa.dot(net.PREV, s.SET*s.SUB) * spa.dot(net.SUB.preconscious, s.ON),
								  self.GW_scale >> net.SUB.attention,
								  .25*s.GET*s.SUB >> net.PREV
							 )

					net.routing_net.labels.append("SET ADD")
					spa.ifmax(net.routing_net.labels[-1], self.BG_bias 
					+ 	spa.dot(net.PREV, s.GET*s.V) * spa.dot(net.G, s.CHAINED_ADD)
					- 	spa.dot(net.GW, s.FIXATE) 	# too soon
					- 	spa.dot(net.PREV, s.GET*s.ADD) # already done
					,
								  1 >> net.ADD.receive,
								  .25*s.SET*s.ADD >> net.PREV,
							 )

					net.routing_net.labels.append("GET ADD")
					spa.ifmax(net.routing_net.labels[-1], self.BG_bias
					+ 	spa.dot(net.PREV, s.SET*s.ADD) * spa.dot(net.ADD.preconscious, s.ON),
								  self.GW_scale >> net.ADD.attention,
								  .25*s.GET*s.ADD >> net.PREV
							 )

					net.routing_net.labels.append("SET M")
					spa.ifmax(net.routing_net.labels[-1], self.BG_bias 
					+ 	spa.dot(net.PREV, s.GET*s.COM)
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

			self.probes = {'BTN': nengo.Probe(net.BTN, synapse=None)}

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
											'out': nengo.Probe(net.GW.output, synapse=synapse),
											'voltages': [nengo.Probe(ens.neurons, 'voltage', synapse=synapse) for ens in net.GW.all_ensembles]}
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

