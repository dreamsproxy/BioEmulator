import numpy as np
import activations
from data_loader import image_dataset, audio
from random import shuffle
from tqdm import trange
from numba import njit

@njit
def _update(inputs, W_i, W_h, W_r, state, learning_rate):
    """
    Compute neuron outputs and update weights using Oja's Rule with interconnections.
    
    - inputs: NumPy array of shape (num_inputs,)
    - W_i: Input-to-neuron weights
    - W_h: Inter-neuron weights
    - W_r: Readout neuron weights
    - state: Previous neuron activations (for recurrent influence)
    """
    inputs = inputs.reshape(-1, 1)  # Ensure column vector

    # Compute neuron activations with recurrent influence
    raw_outputs = np.dot(W_i, inputs) + np.dot(W_h, state.reshape(-1, 1))
    outputs = activations.tanh(raw_outputs)

    # Oja's Rule weight updates
    W_i += learning_rate * (np.outer(outputs, inputs.T) - (outputs**2) * W_i)
    W_h += learning_rate * (np.outer(outputs, outputs.T) - (outputs**2) * W_h)
    W_r += learning_rate * (np.outer(outputs, inputs.T) - (outputs**2) * W_h)  # Update readout weights

    return outputs.flatten(), W_i, W_h, W_r

class OjaNetwork:
    def __init__(self, num_neurons, num_inputs, learning_rate=0.1):
        """
        Multi-Neuron Oja's Rule Network with Interconnections and a Readout Neuron
        
        - num_neurons: Number of neurons
        - num_inputs: Number of input features
        - learning_rate: Oja's Rule learning rate
        """
        self.num_neurons = num_neurons
        self.num_inputs = num_inputs
        self.learning_rate = learning_rate

        # Initialize weights randomly
        self.W_i = np.random.randn(num_neurons, num_inputs) * 0.5
        self.W_h = np.random.randn(num_neurons, num_neurons) * 0.5
        self.W_r = np.random.randn(num_neurons, num_inputs) * 0.5  # Readout weights
        self.state = np.zeros(num_neurons, dtype=np.float32)  # Initial neuron states
        self.W_i = self.W_i.astype(np.float32)
        self.W_h = self.W_h.astype(np.float32)
        self.W_r = self.W_r.astype(np.float32)

    def update(self, inputs):
        """
        Compute neuron outputs and update weights using Oja's Rule with interconnections.
        
        - inputs: NumPy array of shape (num_inputs,)
        """
        self.state, self.W_i, self.W_h, self.W_r = _update(inputs, self.W_i, self.W_h, self.W_r, self.state, self.learning_rate)
        readout_output = np.dot(self.W_r, self.state)  # Compute readout neuron output
        return readout_output

network = OjaNetwork(num_neurons=8, num_inputs=108, learning_rate=0.01)

#datagen = image_dataset((28, 28), (-1.0, 1.0))
#dataset = datagen.flow_from_directory('./datasets/MNIST-JPEG/trainingSet/', n_samples_per_label=128)
#shuffle(dataset)
dataset = audio.load('The Fool.wav')
# Training loop with error tracking
# Reconstruction mode
epochs = 5
for epoch in range(epochs):
    total_error = 0  # Track total error for this epoch
    for i in trange(len(dataset)):
        
        output = network.update(dataset[:, i])

        # Compute squared error
        error = (output - dataset[:, i]) ** 2
        total_error += error

    print(f"\nEpoch {epoch}, Total Error: {total_error}")

# Final Testing after training
print("\nFinal Testing:")
for i in range(len(dataset)):
    inputs = dataset[i][1]
    targets = dataset[i][0]
    output = network.update(inputs)
    print(f"Predicted: {output}, Target: {targets}")
