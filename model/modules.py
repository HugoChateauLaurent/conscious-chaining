import nengo
import nengo_spa as spa
import numpy as np
from nengo.params import Default, IntParam, NumberParam
import itertools

class PartialWTAAssocMem(spa.AssociativeMemory):

	def __init__(
		self,
		threshold,
		input_vocab,
		output_vocab=None,
		mapping=None,
		exceptions=None,
		n_neurons=50,
		label=None,
		seed=None,
		add_to_container=None,
		vocabs=None,
		**selection_net_kwargs
	):
		print(list(exceptions))
		if not hasattr(mapping, "keys"):
			mapping = {k: k for k in mapping}
			
		input_keys = list(mapping.keys())
		exceptions_idx = [[input_keys.index(elt) for elt in e] for e in exceptions]

		selection_net_kwargs["threshold"] = threshold
		super(PartialWTAAssocMem, self).__init__(
			selection_net=PartialWTA,
			exceptions=exceptions_idx,
			input_vocab=input_vocab,
			output_vocab=output_vocab,
			mapping=mapping,
			n_neurons=n_neurons,
			label=label,
			seed=seed,
			add_to_container=add_to_container,
			vocabs=vocabs,
			**selection_net_kwargs
		)

class PartialWTA(spa.networks.selection.Thresholding):
	def __init__(
		self, n_neurons, n_ensembles, exceptions, inhibit_scale=1.0, inhibit_synapse=0.005, **kwargs
	):
		super().__init__(n_neurons, n_ensembles, **kwargs)

		weights = inhibit_scale * (np.eye(n_ensembles) - 1.0)
		for e in exceptions:
			for pair in itertools.combinations(e, 2):
				weights[pair[0],pair[1]] = 0
				weights[pair[1],pair[0]] = 0

		with self:
			nengo.Connection(
				self.thresholded,
				self.input,
				transform=weights,
				synapse=inhibit_synapse,
			)

			
class Delay:
	def __init__(self, dimensions, t_delay, dt=.001):
		self.history = np.zeros((max(1,int(t_delay / dt)), dimensions))

	def step(self, t, x):
		self.history = np.roll(self.history, -1, axis=0)
		self.history[-1] = x
		return self.history[0]

class Processor(spa.Network):

	def __init__(
		self, 
		input_vocab, 
		output_vocab, 
		label,
		sender=True,
		receiver=True,
		**kwargs
		):

		super(Processor, self).__init__(label=label, **kwargs)
		self.sender = sender
		self.receiver = receiver

		with self:

			self.input = spa.Transcode(input_vocab=input_vocab, output_vocab=input_vocab)
			self.preconscious = spa.Transcode(input_vocab=output_vocab, output_vocab=output_vocab)

			if self.receiver:
				self.broadcast = spa.Transcode(input_vocab=input_vocab, output_vocab=input_vocab)
				self.receive = spa.Scalar()
				self.receive_weighting = nengo.networks.Product(200, input_vocab.dimensions, input_magnitude=1.5)
				nengo.Connection(self.receive.output, 
					self.receive_weighting.input_a, 
					transform=[[1]]*input_vocab.dimensions, 
					synapse=None)
				nengo.Connection(self.broadcast.output, 
					self.receive_weighting.input_b,
					synapse=None)
				nengo.Connection(self.receive_weighting.output, 
					self.input.input, 
					synapse=None)

			if self.sender:
				self.attention_weighting = nengo.networks.Product(200, output_vocab.dimensions, input_magnitude=1.5)
				self.attention = spa.Scalar()
				nengo.Connection(self.attention.output, 
					self.attention_weighting.input_a, 
					transform=[[1]]*output_vocab.dimensions,
					synapse=None)
				nengo.Connection(self.preconscious.output, 
					self.attention_weighting.input_b,
					synapse=None)

		self.declare_input(self.input.input, input_vocab)
		self.declare_output(self.preconscious.output, output_vocab)
		if self.sender:
			self.declare_output(self.attention_weighting.output, output_vocab)

class AMProcessor(Processor):

	def __init__(
		self, 
		input_vocab, 
		output_vocab, 
		label,
		AM_mapping,
		feedback=.8, 
		feedback_synapse=.1,
		AM_thr=.1, 
		AM_cls=spa.WTAAssocMem, 
		AM_fn=lambda x:x,
		npd_AM=50, 
		sender=True,
		receiver=True,
		add_ON=True,
		**kwargs
		):

		super(AMProcessor, self).__init__(
			input_vocab=input_vocab,
			output_vocab=output_vocab,
			label=label, 
			sender=sender,
			receiver=receiver,
			**kwargs)
		

		if isinstance(AM_mapping,list):
			AM_mapping = {k:k for k in AM_mapping}

		with self:
	
			# Domain specific processing
			add_ON = "+ON" if add_ON else ""
			self.AM = AM_cls(
				threshold=AM_thr, 
				input_vocab=input_vocab, 
				output_vocab=output_vocab,
				mapping={k:v+add_ON for k,v in AM_mapping.items()},
				function=AM_fn,
				label=self.label+' AM',
				n_neurons = npd_AM
			)
			if feedback:
				nengo.Connection(self.AM.selection.output, 
					self.AM.selection.input,
					transform=feedback, 
					synapse=feedback_synapse)
				
			nengo.Connection(self.input.output, self.AM.input)
			nengo.Connection(self.AM.output, self.preconscious.input, synapse=None)

class DirectCompareProcessor(Processor):

	def __init__(
		self, 
		input_vocab, 
		output_vocab, 
		label,
		s_evidence,
		feedback,
		feedback_synapse,
		reset_period,
		n_neurons_per_dim,
		rng,
		n_samples=1000,
		tau=.05,
		integration_threshold=0,
		input_AM_thr=.1,
		decision_thr=.5, 
		sender=True,
		receiver=True,
		add_ON=True ,
		**kwargs
		):

		super(DirectCompareProcessor, self).__init__(
			input_vocab=input_vocab,
			output_vocab=output_vocab,
			label=label, 
			sender=sender,
			receiver=receiver,
			**kwargs)

		with self:

			n_neurons = n_neurons_per_dim['Ensemble']
			D = input_vocab.dimensions

			self.clean_b = nengo.Node([1]) # constant input for our task (5)
			self.clean_a = spa.State(input_vocab, feedback=feedback, feedback_synapse=feedback_synapse)
			# self.clean_a = spa.ThresholdingAssocMem(
			# 	input_AM_thr,
			# 	input_vocab,
			# 	mapping = ["D2", "D4", "D6", "D8"],
			# 	n_neurons=n_neurons_per_dim['AM']
			# )
			nengo.Connection(self.input.output, self.clean_a.input, synapse=.01)
			# if feedback:
			# 	nengo.Connection(self.clean_a.selection.output, 
			# 		self.clean_a.selection.input,
			# 		transform=feedback, 
			# 		synapse=feedback_synapse)


			train_vectors = np.zeros((n_samples, 2*D))
			train_coords = np.zeros((n_samples, 1))
			xs, ys = rng.choice(range(2,8+1), size=n_samples), rng.choice(range(2,8+1), size=n_samples)
			for i in range(n_samples):
				train_vectors[i, :D] = input_vocab.parse("D"+str(xs[i])).v
				train_vectors[i, D:] = input_vocab.parse("D"+str(ys[i])).v
				train_coords[i, 0] = -1 if xs[i]<ys[i] else 1 if xs[i]>ys[i] else 0
				
			self.combined = nengo.Ensemble(2*D*n_neurons_per_dim['combined'], 2*D)
			nengo.Connection(self.clean_a.output, 
				self.combined[:D], 
				synapse=None)
			nengo.Connection(self.clean_b, 
				self.combined[D:], 
				function=lambda x: x*input_vocab.parse('D5').v, 
				synapse=None)

			self.compared = nengo.Node(size_in=1)
			nengo.Connection(
				self.combined,
				self.compared,
				function=train_coords,
				eval_points=train_vectors,
				scale_eval_points=False,
				synapse=None
			  )

			self.control_signal = spa.networks.selection.Thresholding(
				n_neurons=n_neurons,
				n_ensembles=2,
				threshold=integration_threshold,
				function=lambda x: min(x,1)#x if x<.6 else 1
			)
			nengo.Connection(self.compared, self.control_signal.input[0])
			nengo.Connection(self.compared, self.control_signal.input[1], transform=-1) # if comparison is negative

			
			# Create integrator
			self.integrator = nengo.Ensemble(n_neurons*40, 2, radius=1.5)
			nengo.Connection( # input to the integrator
				self.compared,
				self.integrator[0],
				transform=tau*s_evidence,
				synapse=tau
			  )
			nengo.Connection( # input to the control signal
				self.control_signal.output[0],
				self.integrator[1],
				synapse=tau
			  )

			nengo.Connection( # input to the control signal (if input is negative)
				self.control_signal.output[1],
				self.integrator[1],
				synapse=tau
			  )

			nengo.Connection(self.integrator, self.integrator[0], # feedback
				synapse=tau,
				function=lambda x: x[0]*x[1] # control
			)      

			if reset_period is not None:
				self.integrator_reset = nengo.Node(lambda t: t%reset_period<.2) # reset integrator at the beginning of each trial, during 200 ms (should be at least tau seconds)
				nengo.Connection(
					self.integrator_reset,
					self.integrator.neurons,
					synapse=None,
					transform=-100*np.ones((self.integrator.n_neurons, 1)),
				)
			
			add_ON = "+ON" if add_ON else ""
			self.output = spa.ThresholdingAssocMem(
				decision_thr, 
				output_vocab, 
				mapping={k:k+add_ON for k in ["LESS", "MORE"]},
				function=lambda x: x>0,
				n_neurons = n_neurons_per_dim['AM']

			)
			nengo.Connection(self.integrator[0], self.output.input, transform=[[-v] for v in output_vocab.parse('LESS').v])
			nengo.Connection(self.integrator[0], self.output.input, transform=[[v] for v in output_vocab.parse('MORE').v])

			nengo.Connection(self.output.output, self.preconscious.input, synapse=None)


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