import nengo
import nengo_spa as spa

class Processor(spa.Network):

	def __init__(
		self, 
		input_vocab, 
		output_vocab, 
		label,
		AM_mapping,
		filter_thr,
		filter_mapping=None,
		add_ON=True,
		feedback=.85, 
		feedback_synapse=.1,
		AM_thr=.1, 
		AM_cls=spa.ThresholdingAssocMem, 
		npd_AM=50, 
		broadcast_source=None,
		seed=None, 
		add_to_container=None,
		):


		super(Processor, self).__init__(label, seed, add_to_container, vocabs=None)
		

		if isinstance(AM_mapping,list):
			AM_mapping = {k:k for k in AM_mapping}

		if filter_mapping is None:
			filter_mapping = {k:k for k in AM_mapping.keys()}
		elif isinstance(filter_mapping,list):
			filter_mapping = {k:k for k in filter_mapping}

		self.label = label
		self.sender = broadcast_source in ['GW','processors']

		

		self.input_state = spa.State(input_vocab, feedback=.8, feedback_synapse=.01, label=label+' input')

		# Domain specific processing
		if add_ON:
			add_ON = "+ON"
		else:
			add_ON = ""
		self.AM = AM_cls(
			threshold=AM_thr, 
			input_vocab=input_vocab, 
			mapping={k:v+add_ON for k,v in AM_mapping.items()},
			function=lambda x:x,
			label='ADD',
			n_neurons = npd_AM
		)
		nengo.Connection(self.input_state.output, self.AM.input, synapse=.01)
		self.input_state >> self.AM.input

		if self.sender	:
			self.preconscious = self.AM.output
			self.attention = spa.Scalar()
			1/2 >> self.attention


		if broadcast_source == 'GW':
			self.broadcast_source = spa.Transcode(input_vocab=input_vocab, output_vocab=input_vocab, label=label+' broadcast')
		elif broadcast_source == 'processors':
			self.broadcast_source = self.AM.output

