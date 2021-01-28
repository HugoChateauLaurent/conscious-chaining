import nengo
import nengo_spa as spa
import numpy as np
from nengo.params import Default, IntParam, NumberParam
import itertools
from nengo.networks import InputGatedMemory

s = spa.sym

class GlobalWorkspace(spa.Network):

    def __init__(
        self,
        vocab,
        mappings,
        Theta=.2,
        tau=.01,
        n_neurons=50,
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
            self.AMs = {
                processor : spa.modules.AssociativeMemory(
                    selection_net=GlobalWorkspaceSelection,
                    input_vocab=vocab,
                    output_vocab=vocab,
                    mapping=mappings[processor],
                    n_neurons=n_neurons,
                    Theta=Theta,
                    tau=tau,
                    label=processor.label + ' AM',
                    seed=seed,
                    add_to_container=add_to_container,
                    vocabs=vocabs,
                    **selection_nets_kwargs
                )
                for processor in mappings.keys()
            }

            # create nodes that track the activation of each individual AM
            self.detectors = {}
            for processor in self.AMs.keys():
                self.detectors[processor] = nengo.Node(size_in=1)
                nengo.Connection(
                    self.AMs[processor].selection.thresholding.function, 
                    self.detectors[processor], 
                    transform=np.ones((1,self.AMs[processor].selection.thresholding.function.size_out)),
                    synapse=None
                )

            # create feedback connections
            for processor in self.AMs.keys():
                assert processor.sender
                nengo.Connection(self.AMs[processor].output, processor.topdown_feedback.input, transform=1)

            # create missing lateral inhibitions
            for AM_1 in self.AMs.values():
                for AM_2 in self.AMs.values():
                    if AM_1 != AM_2:
                        nengo.Connection(
                            AM_1.selection.thresholded,
                            AM_2.selection.input,
                            transform=-np.ones((AM_2.selection.input.size_in, AM_1.selection.thresholded.size_out)),
                            synapse=tau,
                        )

            # create GW output
            self.output = nengo.Node(size_in=vocab.dimensions)
            for processor in self.AMs.keys():
                nengo.Connection(self.AMs[processor].output, self.output)

        self.declare_output(self.output, vocab)

class GlobalWorkspaceSelection(spa.networks.selection.Thresholding):

    def __init__(
        self, n_neurons, n_ensembles, Theta, tau=.01, coalitions=(), radius=1, intercept_width=.15, **kwargs
    ):
        super().__init__(
            n_neurons, 
            n_ensembles, 
            0,
            intercept_width,
            lambda x: x>0,
            radius,
            **kwargs)

        I = np.eye(n_ensembles)
        inhibit = - (1 - I)

        for coalition in coalitions:
            inhibit[coalition] = 0
            inhibit[tuple(reversed(coalition))] = 0

        with self:
            nengo.Connection(
                self.thresholded,
                self.input,
                transform=inhibit,
                synapse=tau,
            )

            nengo.Connection(
                self.output,
                self.input,
                transform=Theta * I,
                synapse=tau
            )

class WM(spa.Network):
    def __init__(
        self,
        n_neurons, 
        vocab,
        feedback=1.0, 
        difference_gain=1.0, 
        recurrent_synapse=0.05, 
        difference_synapse=None, 
        **kwargs
    ):
        
        super(WM, self).__init__(**kwargs)
        self.vocab = vocab

        with self:
            self.wm = InputGatedMemory(n_neurons, vocab.dimensions, feedback, difference_gain, recurrent_synapse, difference_synapse)
        
        self.input = self.wm.input
        self.gate = self.wm.gate
        self.reset = self.wm.reset
        self.output = self.wm.output
            
        self.declare_input(self.input, self.vocab)
        self.declare_input(self.gate, None)
        self.declare_input(self.reset, None)
        self.declare_output(self.output, self.vocab)

class Processor(spa.Network):

    def __init__(
        self, 
        input_vocab, 
        output_vocab, 
        label,
        n_neurons_per_dim,
        sender=True,
        receiver=True,
        prediction_in=False,
        prediction_out=False,
        **kwargs
        ):

        super(Processor, self).__init__(label=label, **kwargs)
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.sender = sender
        self.receiver = receiver

        self.prediction_in = prediction_in
        self.prediction_out = prediction_out

        with self:

            self.input = spa.Transcode(input_vocab=input_vocab, output_vocab=input_vocab)

            if self.prediction_in:
                self.processing_input = spa.Transcode(input_vocab=input_vocab, output_vocab=input_vocab)
                self.prediction_in_ens = nengo.Ensemble(
                    n_neurons_per_dim['Ensemble']*self.input_vocab.dimensions,
                    self.input_vocab.dimensions,
                    intercepts=nengo.dists.Uniform(.1, .1)
                )
                nengo.Connection(self.input.output, self.prediction_in_ens, synapse=None)
                nengo.Connection(self.prediction_in_ens, self.processing_input.input)
            else:
                self.processing_input = self.input

            self.processing_output = spa.Transcode(input_vocab=output_vocab, output_vocab=output_vocab)

            if self.prediction_out:
                self.output = spa.Transcode(input_vocab=output_vocab, output_vocab=output_vocab)
                self.prediction_out_ens = nengo.Ensemble(
                    n_neurons_per_dim['Ensemble']*self.output_vocab.dimensions,
                    self.output_vocab.dimensions,
                    intercepts=nengo.dists.Uniform(.1, .1)
                )
                nengo.Connection(self.processing_output.output, self.prediction_out_ens, synapse=None)
                nengo.Connection(self.prediction_out_ens, self.output.input)
            else:
                self.output = self.processing_output


            if self.receiver:
                self.broadcast = spa.Transcode(input_vocab=input_vocab, output_vocab=input_vocab)
                nengo.Connection(self.broadcast.output, self.input.input, synapse=None)

            if self.sender:

                self.preconscious = spa.Transcode(input_vocab=output_vocab, output_vocab=output_vocab)
                nengo.Connection(self.output.output, self.preconscious.input, synapse=None)

                self.topdown_feedback = spa.Transcode(input_vocab=output_vocab, output_vocab=output_vocab)
                nengo.Connection(self.topdown_feedback.output, self.processing_output.input, synapse=None)

                # remove feedback from preconscious
                nengo.Connection(self.topdown_feedback.output, self.preconscious.input, transform=-1, synapse=None)

        self.declare_input(self.input.input, input_vocab)
        self.declare_output(self.output.output, output_vocab)
        if self.receiver:
            self.declare_input(self.broadcast.input, input_vocab)
        if self.sender:
            self.declare_output(self.preconscious.output, output_vocab)

class DirectProcessor(Processor):

    def __init__(
        self, 
        input_vocab, 
        output_vocab, 
        label,
        n_neurons_per_dim,
        sender=True,
        receiver=True,
        **kwargs
        ):

        super(DirectProcessor, self).__init__(input_vocab, output_vocab, label, n_neurons_per_dim, sender, receiver, **kwargs)
        nengo.Connection(self.processing_input.output, self.processing_output.input, synapse=None)

class Prediction:
    def __init__(
      self, 
      source, 
      target,
      rate,
    ):
        self.source = source
        self.target = target
        assert self.source.prediction_out
        assert self.target.prediction_in

        connection = nengo.Connection(
            self.source.prediction_out_ens.neurons,
            self.target.prediction_in_ens.neurons,
            transform=np.zeros((self.target.prediction_in_ens.n_neurons, self.source.prediction_out_ens.n_neurons)),
            learning_rule_type=nengo.BCM(
                learning_rate=rate,
            ),
        )

class AMProcessor(Processor):

    def __init__(
        self, 
        input_vocab, 
        output_vocab, 
        label,
        mapping,
        n_neurons_per_dim,
        feedback=.7,
        threshold=0,
        sender=True,
        receiver=True,
        **kwargs
        ):

        super(AMProcessor, self).__init__(
            input_vocab=input_vocab,
            output_vocab=output_vocab,
            label=label, 
            n_neurons_per_dim=n_neurons_per_dim,
            sender=sender,
            receiver=receiver,
            **kwargs)
        
        with self:
    
            self.AM = spa.ThresholdingAssocMem(
                threshold=threshold, 
                input_vocab=input_vocab, 
                output_vocab=output_vocab,
                mapping=mapping,
                label=self.label+' AM',
                n_neurons = n_neurons_per_dim['AM']
            )
                
            if type(self) == AMProcessor:
                nengo.Connection(self.processing_input.output, self.AM.input)        
                
            nengo.Connection(self.AM.selection.thresholding.output, self.AM.selection.thresholding.input, transform=feedback)     
            nengo.Connection(self.AM.output, self.processing_output.input, synapse=None)

class CompareProcessor(AMProcessor):

    def __init__(
        self, 
        input_vocab, 
        output_vocab, 
        label,
        reset_period,
        reset_duration,
        n_neurons_per_dim,
        rng,
        n_samples=1000,
        tau=.05,
        decision_thr=.75, 
        sender=True,
        receiver=True,
        **kwargs
        ):

        super(CompareProcessor, self).__init__(
            input_vocab=input_vocab,
            output_vocab=output_vocab,
            label=label, 
            mapping=["LESS","MORE"],
            n_neurons_per_dim=n_neurons_per_dim,
            threshold=decision_thr,
            feedback=0,
            sender=sender,
            receiver=receiver,
            **kwargs)


        with self:

            n_neurons = n_neurons_per_dim['Ensemble']
            D = input_vocab.dimensions

            self.a = spa.Transcode(input_vocab=input_vocab, output_vocab=input_vocab)
            self.b = nengo.Node([1]) # constant input for our task (5)

            nengo.Connection(self.input.output, self.a.input, synapse=None)

            train_vectors = np.zeros((n_samples, 2*D))
            train_coords = np.zeros((n_samples, 1))
            xs, ys = rng.choice(range(2,8+1), size=n_samples), rng.choice(range(2,8+1), size=n_samples)
            for i in range(n_samples):

                # avoids small bias when no input
                if rng.random()<.5:
                    train_vectors[i, :D] = 0 if rng.random()<.5 else input_vocab.parse('FIXATE').v
                    train_vectors[i, D:] = input_vocab.parse('D5').v
                    train_coords[i, 0] = 0

                else:
                    train_vectors[i, :D] = input_vocab.parse("D"+str(xs[i])).v
                    train_vectors[i, D:] = input_vocab.parse("D"+str(ys[i])).v
                    train_coords[i, 0] = -1 if xs[i]<ys[i] else 1 if xs[i]>ys[i] else 0
                
            self.combined = nengo.Ensemble(max(10,2*D*n_neurons_per_dim['combined']), 2*D)
            nengo.Connection(self.a.output, self.combined[:D])
            nengo.Connection(self.b, self.combined[D:], 
                transform=[[v] for v in input_vocab.parse('D5').v], 
                synapse=None)

            self.compared = nengo.Node(size_in=1)
            nengo.Connection(
                self.combined,
                self.compared,
                function=train_coords,
                eval_points=train_vectors,
                scale_eval_points=False,
                synapse=tau
              )

            self.integrator = nengo.Ensemble(250, 1)
            nengo.Connection(
                self.compared,
                self.integrator,
                synapse=None
              )
            nengo.Connection(self.integrator, self.integrator, synapse=tau)

            if reset_duration > 0:
                self.integrator_reset = nengo.Node(lambda t: t%reset_period<reset_duration)
                nengo.Connection(
                    self.integrator_reset,
                    self.integrator.neurons,
                    synapse=None,
                    transform=-100*np.ones((self.integrator.n_neurons, 1)),
                )

            nengo.Connection(self.integrator, self.AM.input,
                synapse=.005,
                transform=np.asarray([[-v] for v in output_vocab.parse('LESS').v]) + np.asarray([[v] for v in output_vocab.parse('MORE').v]))
        

class ADDProcessor(AMProcessor):

    def __init__(
        self, 
        vocab, 
        label,
        n_neurons_per_dim,
        rng,
        BG_bias,
        BG_thr,
        feedback,
        sender=True,
        receiver=True,
        **kwargs
        ):

        super(ADDProcessor, self).__init__(
            input_vocab=vocab,
            output_vocab=vocab,
            label=label, 
            n_neurons_per_dim=n_neurons_per_dim,
            mapping=['D2','D4','D6','D8'],
            threshold=0,
            feedback=feedback,
            sender=sender,
            receiver=receiver,
            **kwargs)


        with self:    
            # Domain specific processing
            self.bind = spa.Bind(vocab, n_neurons_per_dim['Bind'])

            with spa.Network() as self.result_controller:
                self.result_controller.labels = []
                with spa.ActionSelection() as self.result_controller.AS:
                    self.result_controller.labels.append("D0 -> D8")
                    spa.ifmax(self.result_controller.labels[-1],
                        BG_bias + spa.dot(s.D0, self.bind.output),
                        s.D8 >> self.AM.input
                    )
                    self.result_controller.labels.append("D10 -> D2")
                    spa.ifmax(self.result_controller.labels[-1],
                        BG_bias + spa.dot(s.D10, self.bind.output),
                        s.D2 >> self.AM.input
                    )
                    self.result_controller.labels.append("no cycle")
                    spa.ifmax(self.result_controller.labels[-1],
                        BG_bias + BG_thr,
                        self.bind >> self.AM.input
                    )
                
            nengo.Connection(self.input.output, self.bind.input_left)

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
                if similarities[i] > self.thr:
                    self.t_last_evt = t
                    return i+1
                        
        return 0
