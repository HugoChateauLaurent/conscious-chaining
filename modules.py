import nengo
import nengo_spa as spa
import numpy as np
from nengo.params import Default, IntParam, NumberParam


class Delay:
    def __init__(self, dimensions, t_delay, dt=.001):
        self.history = np.zeros((max(1,int(t_delay / dt)), dimensions))

    def step(self, t, x):
        self.history = np.roll(self.history, -1)
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
		add_ON=True,
		**kwargs
		):

		super(Processor, self).__init__(label=label, **kwargs)
		

		if isinstance(AM_mapping,list):
			AM_mapping = {k:k for k in AM_mapping}

		self.label = label
		self.sender = sender
		self.receiver = receiver

		with self:

			if not AM_cls is None:
		
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
			


			if not AM_cls is None:
			
				if False: # Adds some proper delay to avoid operation recursion (TODO: remove and control access/broadcast gates separately)
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


			if self.sender:
				self.attention = spa.Scalar()
				.5 >> self.attention

			if self.receiver :
				self.receive = spa.Scalar()


class Button():
	def __init__(self, SP_vectors, trial_length, thr=.5, wait_length=1):
		self.t_last_evt = -100
		self.SP_vectors = SP_vectors
		self.thr = thr
		self.trial_length = trial_length
		self.wait_length = wait_length
	
	def __call__(self,t,x):
		if t//self.trial_length > self.t_last_evt//self.trial_length and t > (t//self.trial_length)*self.trial_length + self.wait_length:
			for i in range(len(self.SP_vectors)):
				similarities = np.dot(self.SP_vectors,x)
				if np.dot(x,self.SP_vectors[i]) > self.thr:
					self.t_last_evt = t
					return i+1
						
		return 0

class DirectCompareProcessor(Processor):

	def __init__(
		self, 
		input_vocab, 
		output_vocab, 
		label,
		s_evidence,
		feedback,
		n_neurons_per_dim,
		n_samples=1000,
		input_AM_thr=.1,
		decision_thr=.25, 
		sender=True,
		receiver=True,
		add_ON=True,
		tau=.05,
		**kwargs
		):

		super(DirectCompareProcessor, self).__init__(
			input_vocab,
			output_vocab, 
			label, 
			None,
			None,
			None,
			None,
			None,
			sender,
			receiver,
			**kwargs)

		with self:

			n_neurons = n_neurons_per_dim['Ensemble']

			D = input_vocab.dimensions

			self.a = spa.Transcode(input_vocab=input_vocab, output_vocab=input_vocab, label="a")
			self.clean_b = nengo.Node([1]) # constant input (5)
			self.input = self.a.input # for our task

			self.clean_a = spa.ThresholdingAssocMem(
				input_AM_thr,
				input_vocab,
				mapping = ["D2", "D4", "D6", "D8"],
				n_neurons=n_neurons_per_dim['AM']
			)
			for ens in self.clean_a.all_ensembles:
				nengo.Connection(ens, ens, transform=feedback, synapse=.05)
			self.a >> self.clean_a

			train_vectors = np.zeros((n_samples, 2*D))
			train_coords = np.zeros((n_samples, 1))
			for i in range(n_samples):
				x = np.random.randint(10) + 1
				y = np.random.randint(10) + 1
				x_SP = input_vocab.parse("D"+str(x)).v
				y_SP = input_vocab.parse("D"+str(y)).v

				train_vectors[i, :D] = input_vocab.parse("D"+str(x)).v
				train_vectors[i, D:] = input_vocab.parse("D"+str(y)).v
				train_coords[i, 0] = -1 if x<y else 1 if x>y else 0
				
			self.compare = nengo.Ensemble(2*D*n_neurons, 2*D)
			nengo.Connection(self.clean_a.output, self.compare[:D])
			nengo.Connection(self.clean_b, self.compare[D:], function=lambda x: x*input_vocab.parse('D5').v)
			
			self.control_signal = nengo.Ensemble(n_neurons*2, 2, radius=1.5)
			for i in range(self.clean_a.selection.thresholding.output.size_out):
				nengo.Connection(self.clean_a.selection.thresholding.output[i], self.control_signal[0])
			nengo.Connection(self.clean_b, self.control_signal[1])
			
			self.controlled = nengo.Ensemble(n_neurons*2, 2, radius=1.5)
			nengo.Connection(
				self.compare,
				self.controlled[0],
				function=train_coords,
				eval_points=train_vectors,
				scale_eval_points=False,
			  )
			nengo.Connection(self.control_signal, self.controlled[1], function=lambda x: x[0]*x[1])
			
			
			# Create accumulator
			self.accumulator = nengo.Ensemble(n_neurons*2, 2, radius=1)
			nengo.Connection(self.accumulator, self.accumulator[0], # recurrent accumulating connection
				function=lambda x: x[0]*x[1], # conrtolled by control signal
				synapse=tau
			)      
			nengo.Connection( # controlled input
				self.controlled,
				self.accumulator[0],
				function=lambda x: x[0]*x[1],
				transform=tau*s_evidence,
				synapse=tau
			  )
			nengo.Connection(self.control_signal, self.accumulator[1], function=lambda x: x[0]*x[1]) # control signal
			
			add_ON = "+SOURCE*"+self.label if add_ON else ""
			self.output = spa.ThresholdingAssocMem(
				decision_thr, 
		        output_vocab, 
		        mapping={k:k+add_ON for k in ["CONTENT*LESS", "CONTENT*MORE"]},
		        function=lambda x: x>0,
		        n_neurons = n_neurons_per_dim['AM']

		    )
			
			nengo.Connection(self.accumulator[0], self.output.input, 
        		function=lambda x: -x*output_vocab.parse("CONTENT*LESS").v+x*output_vocab.parse("CONTENT*MORE").v
    		)

			self.preconscious = self.output.output


		self.declare_input(self.input, input_vocab) # for our task
		self.declare_output(self.output.output, output_vocab)
