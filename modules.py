import nengo
import nengo_spa as spa
import numpy as np

class Delay(object):
	def __init__(self, dimensions, timesteps):
		self.history = np.zeros((timesteps, dimensions))
	def step(self, t, x):
		self.history = np.roll(self.history, -1, 0)
		self.history[-1] = x
		return self.history[0]

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
		for ens in self.AM.all_ensembles:
			nengo.Connection(ens, ens, transform=feedback, synapse=.05)
		
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


class Button():
	def __init__(self, SP_vectors, trial_length, dt=None, thr=.5, focus_length=1):
		self.t_last_evt = -100
		self.SP_vectors = SP_vectors
		self.t_last_step = 0
		self.dt = dt
		self.thr = thr
		self.trial_length = trial_length
		self.focus_length = focus_length
	
	def __call__(self,t,x):
		if not self.dt or t-self.dt > self.t_last_step:
			self.t_last_step = t
			if t//self.trial_length > self.t_last_evt//self.trial_length and t > (t//self.trial_length)*self.trial_length + self.focus_length:
				for i in range(len(self.SP_vectors)):
					similarities = np.dot(self.SP_vectors,x)
					if np.dot(x,self.SP_vectors[i]) > self.thr:
						self.t_last_evt = t
						return i+1
						
		return 0