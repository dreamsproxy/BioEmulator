import LIF
import numpy as np
from utils import array_ops, dataloader
from tqdm import tqdm, trange

class Network:
    def __init__(self, n_neurons, n_inputs, lr, dt, encodings):
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs
        self.lr = lr
        self.encodings = encodings
        # State
        #self.neuron_dict = OrderedDict()
        #for i in range(self.n_neurons):
        #    self.neuron_dict[i] = np.float64(0.0)
        self.neuron_keys = [i for i in range(self.n_neurons)]
        self.input_keys = [i for i in range(self.n_inputs)]
        self.output_keys = [i for i in range(self.n_inputs)] # Data reconstruction mode
        self.weight_matrix = np.random.uniform(-0.9, 0.9, (self.n_neurons, self.n_neurons))
        self.weight_matrix[self.neuron_keys, self.neuron_keys] = np.float64(0.0)

        self.input_weights = np.random.uniform(-0.9, 0.9, (self.n_inputs, self.n_neurons))
        self.output_weights = np.random.uniform(-0.9, 0.9, (self.n_inputs, self.n_neurons)) # Data reconstruction mode
        self.backlog = np.zeros_like(self.weight_matrix, dtype=np.float64)
        self.activation_matrix = np.zeros(self.n_neurons, dtype=np.float64)

        self.neurons = dict()
        self.init_neurons(dt)
    
    def init_neurons(self, dt):
        v_rest = np.random.uniform(-66.0, -64.0, self.n_neurons).astype(np.float64)
        v_reset = v_rest - np.float64(5.0)
        tau = np.random.uniform(19.5, 20.5, self.n_neurons).astype(np.float64)
        self.thresh = np.random.uniform(-55.0, -45.0, self.n_neurons).astype(np.float64)
        self.init_potentials = v_rest + np.float64(2.71)
        for i in range(self.n_neurons):
            if i >= self.n_neurons - self.encodings.shape[0]:
                self.neurons[i] = {
                    'potential': np.float64(-65.0),
                    'dt': np.float64(dt),
                    'tau':np.float64( 20.0),          # Membrane time constant (ms)
                    'v_rest': np.float64(-65.0),      # Resting potential (mV)
                    'v_reset': np.float64(-70.0),     # Reset potential after spike (mV)
                    'v_threshold': np.float64(-55.0)  # Firing threshold (mV)
                }
            else:
                self.neurons[i] = {
                    'potential': self.init_potentials[i],
                    'dt': np.float64(dt),
                    'tau': tau[i],          # Membrane time constant (ms)
                    'v_rest': v_rest[i],      # Resting potential (mV)
                    'v_reset': v_reset[i],     # Reset potential after spike (mV)
                    'v_threshold': self.thresh[i]  # Firing threshold (mV)
                }
        keys = ['potential', 'dt', 'tau', 'v_rest', 'v_reset', 'v_threshold']
        #dtype = [('potential', 'f4'), ('dt', 'f4'), ('tau', 'f4'), ('v_rest', 'f4'), ('v_reset', 'f4'), ('v_threshold', 'f4')]
        #structured_array = np.empty(self.n_neurons, dtype=dtype)
        array = np.empty((self.n_neurons, len(keys)), dtype=np.float64)
        for i in self.neurons:
            neuron = self.neurons[i]
            array[i] = tuple(neuron[key] for key in keys)

        # Access as structured array fields
        self.neurons = array
        del array
        del keys

        self.init_spikes = np.zeros(shape=(self.n_neurons), dtype=np.float64)
        self.init_step = True
        self.post_spikes = self.init_spikes
        self.post_tau = tau
        self.pre_tau = tau
        self.pre_spikes = self.init_spikes
        self.spike_readout = []

        self.global_step_tick = 0
        self.clip_interval = 8
        self.error_thresholds = np.linspace(1.0, 0.1, num=10)

    def step(self, stream):
        stream = stream.flatten()
        # Use the selected activation function
        io_outputs = LIF.step(stream)

        in_feed = np.zeros_like(self.input_weights)
        for i, x in enumerate(io_outputs):
            in_feed[i, :] = x * self.input_weights[i, :]

        out_feed = np.zeros_like(self.output_weights)
        for i, x in enumerate(io_outputs):
            out_feed[i, :] = x * self.output_weights[i, :]

        io_feed = np.vstack([in_feed, out_feed])
        # Feed input neuron's outputs into network
        hidden_outputs = np.zeros_like(self.weight_matrix, dtype=np.float64)
        for nid in self.neuron_keys:
            if self.backlog.sum() != np.float64(0.0):
                inputs = np.sum(io_feed[:, nid]) + np.sum(self.backlog[nid, :])
                inputs = inputs / np.mean(inputs)
            else:
                inputs = np.sum(io_feed[:, nid]) / np.mean(io_feed[:, nid])
            y = np.tanh(inputs)
            self.activation_matrix[nid] = y
            #self.neuron_dict[nid] = y
            hidden_outputs[nid, :] = self.weight_matrix[nid] * y
        self.backlog = hidden_outputs.copy()
        self.input_weights += np.outer(io_outputs, self.activation_matrix) * self.lr
        self.input_weights *= (1 - self.lr * 0.01)
        self.input_weights = array_ops.minmax_scale(self.input_weights, -0.5, 0.5, axis=1)
        
        self.output_weights += np.outer(io_outputs, self.activation_matrix) * self.lr
        self.output_weights *= (1 - self.lr * 0.01)
        self.output_weights = array_ops.minmax_scale(self.output_weights, -0.5, 0.5, axis=1)
        #print(self.output_weights.min(), self.output_weights.max(), self.output_weights.std())

    def adjust_weights(self, logging=False, scale_every=10, step_count=0):
        """
        Adjust weights with optional periodic scaling.
        Adaptive weight decay that decreases over epochs.
        """
        self.weight_matrix += np.outer(self.activation_matrix, self.activation_matrix) * self.lr
        self.weight_matrix[self.neuron_keys, self.neuron_keys] = 0.0
        decay_rate = 0.01
        self.weight_matrix *= (1 - self.lr * decay_rate)

        # Scale only every `scale_every` steps
        if step_count % scale_every == 0:
            self.weight_matrix = array_ops.minmax_scale(self.weight_matrix, -0.99, 0.99)

        if logging:
            return self.weight_matrix.copy()
        else:
            return None
    
    def reconstruct_old(self, stream):
        result = np.empty_like(stream, dtype=np.float64)
        for ri, r in enumerate(stream):
            r = r.flatten()
            io = LIF.step(r)
            io_feed = np.zeros_like(self.input_weights)
            for i, x in enumerate(io):
                io_feed[i, :] = x * self.input_weights[i, :]
            # Feed input neuron's outputs into network
            hidden_outputs = []
            for nid in self.neuron_keys:
                inputs = np.sum(io_feed[:, nid])
                #if backlog.sum() != np.float64(0.0):
                #    inputs = np.mean([inputs, np.sum(backlog[nid, :])])
                y = np.tanh(inputs)
                hidden_outputs.append(y)
                #hidden_outputs[nid, :] = self.weight_matrix[nid] * y
            # Feed hidden outputs to output neurons
            r_out = np.empty(len(self.output_keys), dtype=np.float64)
            for oid in self.output_keys:
                y = np.tanh(np.mean(self.output_weights[oid, :] * np.array(hidden_outputs, dtype=np.float64)))
                #y = tanh(x)
                r_out[oid] = y
            #print(r_out.shape)
            r_out = array_ops.minmax_scale(r_out.reshape((32, 32)), -1, 1)
            #r_out = r_out.reshape((32, 32))
            result[ri] = r_out
        return result

    def reconstruct(self, stream):
        result = np.empty_like(stream, dtype=np.float64)
        for i, r in enumerate(stream):
            r = r.flatten()
            io_outputs = LIF.step(r)
            step_logs = []
            for j in range(256):
                # Pass through network weights
                hidden_outputs = np.dot(io_outputs, self.input_weights)
                hidden_outputs = LIF.step(hidden_outputs)
                
                output_recon = np.dot(hidden_outputs, self.output_weights.T)
                output_recon = LIF.step(output_recon)
                step_logs.append(output_recon.reshape((64, 64, 1)))
            step_logs = np.array(step_logs)
            step_logs = np.mean(step_logs, axis=0)
            result[i] = array_ops.minmax_scale(step_logs, 0, 255, -1)
        return result

import matplotlib
matplotlib.use('tkagg')
net = Network(512, 64*64, 0.001)
#stream = dataloader.load_image(
#    './datasets/1-119795204_p1_master1200.png',
#    (128, 128))
#stream = dataloader.load_gif('./datasets/gif/4c9ccd3fc92d236f3fcdcb4f5553938c.gif', (32, 32))
stream = dataloader.load_video('./lichen.mp4')

from matplotlib import pyplot as plt
weight_log = []

for i in trange(stream.shape[0]):
    epoch_weights = []
    for _ in range(8):
        net.step(stream[i])
        epoch_weights.append(net.adjust_weights(logging=True))
    
weight_log.append(epoch_weights[-1])

result = net.reconstruct(stream)
for i, rf in enumerate(result):
    print(result.max(), result.min())
    rfig, rax = plt.subplots(1, 2)
    rax[0].imshow(stream[i], cmap='gray')
    rax[1].imshow(rf, cmap='gray')
    rfig.savefig(f'./output/{i}.png', dpi=300)
    plt.close()

for i, f in enumerate(weight_log):
    if i % 2 == 0 or i == stream.shape[1]-1:
        fig = plt.figure(figsize=(8, 8), frameon=True, dpi=300)
        plt.imshow(f, cmap='viridis')
        plt.savefig(f'./gif/{i}.png')
        plt.close()