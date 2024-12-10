import numpy as np
from sklearn.preprocessing import MinMaxScaler

class WeightMatrix:
    def __init__(self, num_neurons, pre_alpha, post_alpha, init_type='glorot_uniform'):
        self.a_pre = pre_alpha
        self.a_post = post_alpha
        self.num_neurons = num_neurons
        self.weights = self._initialize_weights(init_type)
        self.hebb_weights = np.random.uniform(0.0, 1.0, (self.num_neurons, self.num_neurons))
        self.encoding_weights = np.random.uniform(0.25, 0.75, (self.num_neurons, 2))

    def _initialize_weights(self, init_type):
        limit = np.sqrt(6 / (self.num_neurons + self.num_neurons))
        return np.random.uniform(-limit, limit, (self.num_neurons, self.num_neurons))

    def compute_spikes(self, potentials, threshold):
        """
        Computes spike values based on the membrane potentials and thresholds.
        """
        return np.maximum(0, potentials - threshold)

    def update_weights_hebb(self, post_spikes) -> None:
        hebb_increase = np.where(post_spikes > 0.0)
        hebb_decrease = np.where(post_spikes == 0.0)
        self.hebb_weights[hebb_increase] += self.a_post
        self.hebb_weights[hebb_decrease] -= self.a_pre
        self.hebb_weights = np.clip(self.hebb_weights, 1e-16, 1.0)

    def update_weights_stdp(self, pre_spikes, post_spikes, tau_pre, tau_post) -> None:
        # Compute timing differences
        delta_t = np.outer(post_spikes, pre_spikes) - np.outer(pre_spikes, post_spikes)
        # Compute STDP updates proportional to spike strengths
        stdp_update = np.where(
            delta_t > 0,
            self.a_pre * post_spikes[:, None] * np.exp(-delta_t / tau_pre),  # Potentiation
            -self.a_post * pre_spikes[None, :] * np.exp(delta_t / tau_post)  # Depression
        )
        
        self.weights += stdp_update
        self.weights = np.clip(self.weights, 1e-16, 1)

    def update_weights_combined(self, pre_spikes, post_spikes, tau_pre, tau_post):
        self.update_weights_stdp(pre_spikes, post_spikes, tau_pre, tau_post)
        self.update_weights_hebb(post_spikes)
        self.weights = np.mean([self.weights, self.hebb_weights], axis=0)
        self.weights = np.clip(self.weights, 1e-16, 1)

    def update_encodings(self, encoding, spikes):
        self.encoding_weights
        enc_increase = np.where(spikes > 0.0)
        enc_decrease = np.where(spikes == 0.0)

        self.encoding_weights[enc_increase] += self.a_post
        self.encoding_weights[enc_decrease] -= self.a_pre
        self.encoding_weights = np.clip(self.encoding_weights, 0.0, 1.0)
        self.encoding_weights = (self.encoding_weights - self.encoding_weights.min()) / (self.encoding_weights.max() - self.encoding_weights.min())

    def propagate_signals(self, spikes, method='sum'):
        """
        Propagates signals through the network based on current weights.
        
        Args:
            spikes (np.ndarray): Current spikes (binary array).
        
        Returns:
            np.ndarray: Signals propagated to all neurons.
        """
        inputs = np.dot(self.weights, spikes)
        if method == 'sum':
            return inputs
        elif method == 'mean':
            return inputs / self.weights.shape[1]  # Divide by number of connections
        elif method == 'weighted':
            row_sums = self.weights.sum(axis=1, keepdims=True) + 1e-8  # Avoid division by zero
            return inputs / row_sums.squeeze()
        else:
            raise ValueError("Invalid method. Choose from 'sum', 'mean', or 'weighted'.")
    
    def prune_weights(self, threshold=0.01):
        """
        Prunes weak weights below a certain threshold for efficiency.
        
        Args:
            threshold (float): Minimum value for weights to remain non-zero.
        """
        self.weights[self.weights < threshold] = 0.0
    
    def get_weights(self):
        """
        Returns the current weight matrix.
        
        Returns:
            np.ndarray: Current weight matrix.
        """
        return self.weights

