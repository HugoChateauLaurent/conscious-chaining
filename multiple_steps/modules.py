import nengo
import nengo_spa as spa
import numpy as np
from nengo.params import Default, IntParam, NumberParam
import itertools

class AbstractProcessor(spa.Network):

	def __init__(
		self, 
		input_vocab, 
		output_vocab, 
		label,
		sender=True,
		receiver=True,
		**kwargs
		):

		super(AbstractProcessor, self).__init__(label=label, **kwargs)
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

class AMProcessor(AbstractProcessor):

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

class Button():
	def __init__(self, SP_vectors, thr=.5, end_sim=True):
		self.count = 0
		self.SP_vectors = SP_vectors
		self.thr = thr
	
	def __call__(self,t,x):

		if self.count == 0:
			# assert self.count <= 500 # stop simulation if already answered
			for i in range(len(self.SP_vectors)):
				similarities = np.dot(self.SP_vectors, x)
				if np.dot(x,self.SP_vectors[i]) > self.thr:
					self.count += 1
					return i+1
				
		return 0