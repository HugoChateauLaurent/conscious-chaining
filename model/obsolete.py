class GlobalWorkspace(spa.Network):

    def __init__(
        self,
        vocab,
        mappings,
        Theta=.2,
        n_neurons=50,
        inhibit_scale=1.0,
        inhibit_synapse=0.005,
        label="Global Workspace",
        seed=None,
        add_to_container=None,
        vocabs=None,
        **selection_nets_kwargs
    ):

        super(GlobalWorkspace, self).__init__(
            label=label, seed=seed, add_to_container=add_to_container, vocabs=vocabs
        )

        with self:

            # create winner take all networks
            self.WTAs = {
                processor : spa.WTAAssocMem(
                    threshold=0,
                    input_vocab=vocab,
                    output_vocab=vocab,
                    mapping=mappings[processor],
                    n_neurons=n_neurons,
                    function=lambda x: x>0,
                    label=processor.label + ' WTA',
                    seed=seed,
                    add_to_container=add_to_container,
                    vocabs=vocabs,
                    **selection_nets_kwargs
                )
                for processor in mappings.keys()
            }

            # create nodes that track the activation of each individual WTA
            self.detectors = {}
            for processor in self.WTAs.keys():
                self.detectors[processor] = nengo.Node(size_in=1)
                nengo.Connection(
                    self.WTAs[processor].selection.thresholding.function, 
                    self.detectors[processor], 
                    transform=np.ones((1,self.WTAs[processor].selection.thresholding.function.size_out))
                )
                # self.declare_output(self.detectors[processor], None)
            self.detectors = {
                processor: nengo.Node(size_in=1)
                for processor in mappings.keys()
            }

            # create missing lateral inhibitions
            for WTA_1 in self.WTAs.values():
                for WTA_2 in self.WTAs.values():
                    if WTA_1 != WTA_2:
                        nengo.Connection(
                            WTA_1.selection.thresholded,
                            WTA_2.selection.input,
                            transform=-inhibit_scale * np.ones((WTA_2.selection.input.size_in, WTA_1.selection.thresholded.size_out)),
                            synapse=inhibit_synapse,
                        )

            # create feedback connections
            for processor in self.WTAs.keys():
                assert processor.sender
                nengo.Connection(self.WTAs[processor].output, processor.topdown_feedback.input, transform=.25)

            # create GW output
            self.output = nengo.Node(size_in=vocab.dimensions)
            for processor in self.WTAs.keys():
                nengo.Connection(self.WTAs[processor].output, self.output)

        self.declare_output(self.output, vocab)


class Prediction:
    def __init__(
      self, 
      source, 
      target,
      n_neurons_per_dim,
      voja_rate,
      pes_rate,
      T_learning=np.inf
    ):
        self.source = source
        self.target = target
        keys = self.source.output_vocab.vectors

        assert self.source.output_vocab.dimensions == self.target.input_vocab.dimensions
        D = self.source.output_vocab.dimensions

        # create output state in the source
        if not hasattr(self.source, 'prediction_source'):
            self.source.prediction_source = spa.State(self.source.output_vocab, neurons_per_dimension=n_neurons_per_dim['State'], represent_cc_identity=False)
            nengo.Connection(self.source.output.output, self.source.prediction_source.input, synapse=None)

        # create ensembles that become selective to their inputs with Voja
        subdimensions = self.source.prediction_source.subdimensions
        intercepts = [(np.dot(keys[:,i*subdimensions:(i+1)*subdimensions], keys[:,i*subdimensions:(i+1)*subdimensions].T) - np.eye(len(keys))).flatten().max() for i in range(D//subdimensions)]
        self.encoding_ensembles = [
            nengo.Ensemble(
                n_neurons_per_dim['Ensemble']*subdimensions, 
                subdimensions, 
                intercepts=[intercepts[i]]*n_neurons_per_dim['Ensemble']*subdimensions
            )
            for i in range(D//subdimensions)
        ]
        voja = nengo.Voja(learning_rate=voja_rate, post_synapse=None)
        self.encoding_connections = [
            nengo.Connection(
                self.source.prediction_source.all_ensembles[i], 
                self.encoding_ensembles[i], 
                learning_rule_type=voja
            )
            for i in range(len(self.source.prediction_source.all_ensembles))
        ]

        # Learn the decoders/values, initialized to a null function
        pes = nengo.PES(pes_rate)
        self.prediction = spa.Transcode(input_vocab=self.target.input_vocab, output_vocab=self.target.input_vocab)
        self.decoding_connections = [
            nengo.Connection(
                self.encoding_ensembles[i], 
                self.prediction.input[i*subdimensions:(i+1)*subdimensions],
                learning_rule_type=pes,
                function=lambda x: np.zeros(subdimensions),
            )
            for i in range(D//subdimensions)
        ]
        nengo.Connection(self.prediction.output, self.target.prediction_input.input, synapse=None)

        # Create the error nodes. Learning is inhibited when there is no information in source or ground truth
        self.norm_product = nengo.Node(lambda t,x: np.linalg.norm(x[:D])*np.linalg.norm(x[D:]), size_in=2*D)
        nengo.Connection(self.source.prediction_source.output, self.norm_product[:D])
        nengo.Connection(self.target.true_input.output, self.norm_product[D:])
        self.error = nengo.Node(lambda t,x: x[:-1]*x[-1] if t<T_learning else 0, size_in=self.target.input_vocab.dimensions+1, size_out=self.target.input_vocab.dimensions)
        for i in range(D//subdimensions):
            nengo.Connection(self.norm_product, self.error[-1], synapse=None)
            nengo.Connection(self.target.true_input.output[i*subdimensions:(i+1)*subdimensions], self.error[i*subdimensions:(i+1)*subdimensions], transform=-1, synapse=None)
            nengo.Connection(self.prediction.output[i*subdimensions:(i+1)*subdimensions], self.error[i*subdimensions:(i+1)*subdimensions], synapse=None)
            nengo.Connection(self.error[i*subdimensions:(i+1)*subdimensions], self.decoding_connections[i].learning_rule, synapse=None)