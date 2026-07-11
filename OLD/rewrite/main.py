import numpy as np
import tqdm
import os
import cv2
from glob import glob
from matplotlib import pyplot as plt
from LIF import step
from LIF import default_params as default_neuron_params
from collections import OrderedDict
from sys import getsizeof
from utils import unit_conversion

class w_init:
    def __init__(self):
        pass

    @staticmethod
    def glorot(size:int) -> np.ndarray:
        '''
        This is useful when using sigmoid or tanh-like
        activation, but it's not ideal for SNNs.
        
        It ensures that inputs do not explode or vanish by scaling the weights.
        '''
        weights = np.random.uniform(
            -1.0 / np.sqrt(size), 
            1.0 / np.sqrt(size), 
            (size, size), dtype=np.float32)
        return weights
    
    @staticmethod
    def normal(size:int) -> np.ndarray:
        '''
        Bio-plausibility

        Instead of uniform initialization, Gaussian-distributed
        weights can be used, with small positive values and a
        mean close to 0.
        '''
        weights = np.random.normal(
            0, 1.0 / np.sqrt(size),
            (size, size)
        )
        weights = weights.astype(np.float32)
        return weights

    @staticmethod
    def sparsify(weights:np.ndarray, size:int, connectivity:float = 0.2) -> np.ndarray:
        size = weights.shape
        mask = np.random.rand(size[0], size[1]).astype(np.float32) < connectivity
        return weights * mask  # Zero out some weights


class neural_network:
    def __init__(self, n_inputs:int, n_outputs:int, n_hidden:int, lr:np.float32, decay:np.float32, size:np.float32):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden = n_hidden
        self.lr = lr
        self.decay = decay
        self.size_mm = size
        self.total_neurons = self.n_inputs + self.n_outputs + self.n_hidden
        self.weights = self.create_weight_matrix()
        self.neuron_params = self.create_neuron_params()
        #self.neural_matter = self.create_neural_matter()

        self.spike_times = {i: [] for i in range(self.total_neurons)}

        pass

    def create_weight_matrix(self, method:str='sparse'):
        weights = w_init.normal(self.total_neurons)
        weights = w_init.sparsify(weights, self.total_neurons, 0.5)
        weights[np.where(np.eye(self.total_neurons, self.total_neurons,dtype=np.float32) == 1.0)] = np.nan
        weights[:self.n_inputs, :self.n_inputs] = np.nan
        weights[1-(self.n_outputs-1):, 1-(self.n_outputs-1):] = np.nan
        weights[:self.n_inputs, 1-(self.n_outputs-1):] = np.nan
        weights[1-(self.n_outputs-1):, :self.n_inputs] = np.nan
        #plt.imshow(weights)
        #plt.show()
        #print(weights)
        return weights

    def create_neuron_params(self):
        params = OrderedDict({})
        for i in range(self.total_neurons):
            params[i] = default_neuron_params()
        return params

    def create_neural_matter(self):
        matter = np.zeros((self.total_neurons, self.total_neurons, self.total_neurons), dtype=np.uint)
        indices = np.random.randint(0, self.total_neurons, (3, self.total_neurons))
        matter[indices] = np.uint(1)
        #print(indices.shape)
        print(getsizeof(matter))

    def stdp_update(self, weights, spike_times, A_plus=0.01, A_minus=0.012, tau_plus=20, tau_minus=20):
        """
        Applies STDP to update synaptic weights based on spike timing.

        Args:
            weights: Synaptic weight matrix.
            spike_times: Dictionary of neuron spike times.
        
        Returns:
            Updated weight matrix.
        """
        for pre in range(self.total_neurons):
            for post in range(self.total_neurons):
                if pre == post or np.isnan(weights[pre, post]):
                    continue  # Skip self-connections or invalid weights

                if spike_times[pre] and spike_times[post]:  # Check if neurons spiked
                    delta_t = np.array(spike_times[post])[:, None] - np.array(spike_times[pre])

                    delta_w = np.where(delta_t > 0, 
                                    A_plus * np.exp(-delta_t / tau_plus), 
                                    -A_minus * np.exp(delta_t / tau_minus))
                    
                    weights[pre, post] += np.sum(delta_w)  # Apply STDP update

        weights = np.clip(weights, 0, 1)  # Keep weights in valid range
        return weights

    def run(self, timesteps=1000):
        """
        Runs the neural network simulation.
        
        Args:
            timesteps: Number of time steps to simulate.
        """
        img = crop_center_square(cv2.imread('./data/375c3ea1278d32eb4f39dcbf8d82d09d.jpg', cv2.IMREAD_GRAYSCALE))
        img = cv2.resize(img, (64, 64))
        img = img.flatten()
        img = img.astype(np.float32)
        img /= 255.0
        img = img.reshape(-1, 1)
        signals = img * self.weights[:img.shape[0]]
        for t in range(timesteps):
            for i in range(self.n_inputs):
                v, I_syn, spike, refrac = step(self.neuron_params[i], self.weights[i, :], signals[i], 0)  # Update neuron
                self.neuron_params[i]['p'] = v
                self.neuron_params[i]['refrac'] = refrac
                if spike:
                    self.spike_times[i].append(t)  # Log spike event

            if t % 10 == 0:  # Apply STDP every 10 timesteps
                self.weights = self.stdp_update(self.weights, self.spike_times)

def crop_center_square(image: np.ndarray) -> np.ndarray:
    """
    Crops a square from the center of the image, using the smaller dimension as both width and height.
    
    :param image: Input image as a NumPy array.
    :return: Cropped square image as a NumPy array.
    """
    height, width = image.shape
    size = min(height, width)
    
    center_x, center_y = width // 2, height // 2
    x1, y1 = center_x - size // 2, center_y - size // 2
    x2, y2 = x1 + size, y1 + size
    
    return image[y1:y2, x1:x2]

net = neural_network(64*64, 64, 1, 0.001, 0.01, 1)
paths = glob('./data/**')
stream = []
net.run(10000)
#net.run()