import numpy as np
from tanh import tanh_activation as activate
from utils import array_ops, dataloader
from tqdm import tqdm

class network:
    def __init__(self, n_neurons, n_inputs, lr):
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs
        self.lr = lr
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

    def step(self, stream):
        stream = stream.flatten()
        io_outputs = activate(stream)

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
        #self.backlog = hidden_outputs.copy()
        self.input_weights += np.outer(io_outputs, self.activation_matrix) * self.lr
        self.input_weights *= (1 - self.lr * 0.01)
        self.input_weights = array_ops.minmax_scale(self.input_weights, -0.5, 0.5, axis=1)
        
        self.output_weights += np.outer(io_outputs, self.activation_matrix) * self.lr
        self.output_weights *= (1 - self.lr * 0.01)
        self.output_weights = array_ops.minmax_scale(self.output_weights, -0.5, 0.5, axis=1)
        #print(self.output_weights.min(), self.output_weights.max(), self.output_weights.std())

    def adjust_weights(self, logging=False):
        self.weight_matrix += np.outer(self.activation_matrix, self.activation_matrix)*self.lr
        self.weight_matrix[self.neuron_keys, self.neuron_keys] = np.float64(0.0)
        #self.weight_matrix = np.clip(self.weight_matrix, -0.99, 0.99, out=self.weight_matrix)
        #if self.weight_matrix.max() > np.float64(0.999) or self.weight_matrix.min() < np.float64(-0.999):
        #self.weight_matrix = array_ops.minmax_scale(self.weight_matrix, -0.99, 0.99, axis=-1)
        decay_rate = 0.01  # Adjust as needed
        self.weight_matrix *= (1 - self.lr * decay_rate)
        self.weight_matrix[self.neuron_keys, self.neuron_keys] = np.float64(0.0)
        if logging:
            return self.weight_matrix.copy()
        else:
            return None
    
    def reconstruct(self, stream):
        result = np.empty_like(stream, dtype=np.float64)
        backlog = np.zeros_like(self.backlog, dtype=np.float64)
        for ri, r in enumerate(stream):
            r = r.flatten()
            io = activate(r)
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
                #y = activate(x)
                r_out[oid] = y
            #print(r_out.shape)
            r_out = array_ops.minmax_scale(r_out.reshape((32, 32)), -1, 1)
            #r_out = r_out.reshape((32, 32))
            result[ri] = r_out
        return result

import matplotlib
matplotlib.use('tkagg')
net = network(512, 32*32, 0.001)
#stream = dataloader.load_image(
#    './datasets/1-119795204_p1_master1200.png',
#    (128, 128))
stream = dataloader.load_gif('./datasets/gif/4c9ccd3fc92d236f3fcdcb4f5553938c.gif', (32, 32))

from matplotlib import pyplot as plt
weight_log = []
for i in range(1):
    epoch_weights = []
    for i in tqdm(range(stream.shape[0])):
        for _ in range(10):
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