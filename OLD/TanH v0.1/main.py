import activations
import numpy as np
from utils import array_ops, dataloader
from tqdm import tqdm, trange

class network:
    def __init__(self, n_neurons, n_inputs, n_enc, lr):
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs + n_enc
        self.n_enc = n_enc
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
        print(self.input_weights.shape)

    def step(self, stream, encodings):
        stream = stream.flatten()
        stream = np.append(stream, encodings)
        # Use the selected activation function
        io_outputs = activations.tanh(stream)

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
            inputs = (io_feed[:, nid] - np.mean(io_feed[:, nid])) / (np.std(io_feed[:, nid]) + 1e-8)  # Normalize input range
            print(inputs.shape)
            print(self.backlog.shape)
            if self.backlog.sum() >= np.float64(1e-5) or self.backlog.sum() <= np.float64(-1e-5):
                inputs = np.mean([inputs, np.sum(self.backlog[nid, :])])
            y = np.mean(np.tanh(inputs))
            self.activation_matrix[nid] = y
            #self.neuron_dict[nid] = y
            hidden_outputs[nid, :] = self.weight_matrix[nid] * y

        self.backlog = hidden_outputs.copy()

        # Do recall here:
        io_feed = np.zeros_like(self.input_weights)
        for i, x in enumerate(io_outputs):
            io_feed[i, :] = x * self.input_weights[i, :]
        # Feed input neuron's outputs into network
        hidden_outputs = []
        for nid in self.neuron_keys:
            #inputs = np.mean(io_feed[:, nid])
            inputs = (io_feed[:, nid] - np.mean(io_feed[:, nid])) / (np.std(io_feed[:, nid]) + 1e-8)  # Normalize input range
            if self.backlog.sum() >= np.float64(1e-5) or self.backlog.sum() <= np.float64(-1e-5):
                inputs = np.mean([inputs, np.sum(self.backlog[nid, :])])
            y = np.mean(np.tanh(inputs))
            hidden_outputs.append(y)
        hidden_outputs = np.array(hidden_outputs)
        # Feed hidden outputs to output neurons
        r_out = np.empty(len(self.output_keys), dtype=np.float64)
        for oid in self.output_keys:
            r_out[oid] = np.tanh(np.mean(self.output_weights[oid, :] * np.array(hidden_outputs, dtype=np.float64)))

        # Compute error for weight updates
        diff = r_out - stream
        hidden_errors = diff @ self.output_weights
        # Hebbian-like weight update
        delta_w = self.lr * np.outer(io_outputs, hidden_errors)
        self.input_weights += delta_w * np.sign(io_outputs).reshape(-1, 1)
        self.output_weights += delta_w * np.sign(hidden_errors)

        #self.input_weights += np.outer(io_outputs, self.activation_matrix) * self.lr
        #self.output_weights += np.outer(io_outputs, self.activation_matrix) * self.lr

        # Gradually decay backlog instead of resetting
        self.backlog *= 0.9
        # Backpropagate error from output to hidden layer
        #hidden_errors = diff @ self.output_weights  # Shape: (n_hidden,)

        # Update output weights (mapping hidden -> output)
        #self.output_weights -= self.lr * diff[:, None] * self.activation_matrix[None, :]
        #self.input_weights -= self.lr * np.outer(io_outputs, hidden_errors)  # Shape: (n_inputs, n_hidden)

    def adjust_weights(self, logging=False):
        """
        Adjust weights with optional periodic scaling.
        Adaptive weight decay that decreases over epochs.
        """
        decay_rate = 0.05

        self.weight_matrix += np.outer(self.activation_matrix, self.activation_matrix) * self.lr
        self.weight_matrix[self.neuron_keys, self.neuron_keys] = 0.0
        #self.weight_matrix *= (1 - self.lr * decay_rate)

        #self.input_weights *= (1 - self.lr * 0.05)
        #self.output_weights *= (1 - self.lr * 0.05)

        self.weight_matrix *= 1 / (1 + self.lr * 0.01)
        self.input_weights *= 1 / (1 + self.lr * 0.01)
        self.output_weights *= 1 / (1 + self.lr * 0.01)

        self.weight_matrix -= np.mean(self.weight_matrix)
        self.input_weights -= np.mean(self.input_weights)
        self.output_weights -= np.mean(self.output_weights)

        if logging:
            return self.weight_matrix.copy()
        else:
            return None
    
    def reconstruct_old(self, stream, encodings):
        result = np.empty_like(stream, dtype=np.float64)
        enc_out = np.empty_like(encodings, dtype=np.float64)
        for ri, r in enumerate(stream):
            r = r.flatten()
            r = np.append(r, encodings[ri])
            io = activations.tanh(r)
            io_feed = np.zeros_like(self.input_weights)
            for i, x in enumerate(io):
                io_feed[i, :] = x * self.input_weights[i, :]
            # Feed input neuron's outputs into network
            hidden_outputs = []
            for nid in self.neuron_keys:
                inputs = np.mean(io_feed[:, nid])
                #if self.backlog.sum() != np.float64(0.0):
                #    inputs = np.mean([inputs, np.sum(self.backlog[nid, :])])
                y = np.tanh(inputs)
                hidden_outputs.append(y)
            # Feed hidden outputs to output neurons
            r_out = np.empty(len(self.output_keys), dtype=np.float64)
            for oid in self.output_keys:
                r_out[oid] = np.tanh(np.mean(self.output_weights[oid, :] * np.array(hidden_outputs, dtype=np.float64)))
            enc_out = r_out[-self.n_enc:]
            r_out = r_out[:-self.n_enc]
            r_out = r_out.reshape(8, 8, 1)

            result[ri] = r_out
            print(result.shape)
            print(enc_out.shape)
        return result, enc_out

import matplotlib
matplotlib.use('tkagg')
from matplotlib import pyplot as plt
#stream = dataloader.load_image(
#    './datasets/1-119795204_p1_master1200.png',
#    (128, 128))
#stream, encodings = dataloader.load_gif('./dec79cfad5bcf0efc449d92eef9db641.gif', (64, 64))
stream, encodings = dataloader.load_video('./lichen.mp4')
#print(stream)
#raise
net = network(512, 8*8, len(encodings), 0.0271)
weight_log = []

for _ in trange(10):
    epoch_weights = []
    for i in range(stream.shape[0]):
        net.step(stream[i], encodings[i])
        epoch_weights.append(net.adjust_weights(logging=True))
    #net.backlog = np.zeros_like(net.backlog, dtype=np.float64)
    weight_log.append(epoch_weights[-1])

result, enc = net.reconstruct_old(stream, encodings)
for i, rf in enumerate(result):
    rfig, rax = plt.subplots(1, 3)
    rax[0].imshow(stream[i], cmap='gray')
    rax[1].imshow(rf, cmap='gray')
    rax[2].imshow(np.abs(stream[i]) - np.abs(rf), cmap='viridis')
    rax[2].title.set_text(str(encodings[i]))
    rfig.savefig(f'./output/{i}.png', dpi=300)
    plt.close()

for i, f in enumerate(weight_log):
    fig = plt.figure(figsize=(8, 8), frameon=True, dpi=300)
    plt.imshow(f, cmap='viridis')
    plt.savefig(f'./gif/{i}.png')
    plt.close()