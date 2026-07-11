import LIF
from WeightMatrix import WeightMatrix
import numpy as np
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm
from numba import njit

@njit(parallel=True)
def normalize(arr):
    arr = (arr-np.min(arr))/(np.max(arr)-np.min(arr))
    return arr

class DataLoader:
    def __init__(self, mode='image') -> None:
        pass

    def load_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float64)
        # Crop square
        img = img[:640, :640]
        img = cv2.resize(img, (64, 64))
        img /= 255.0
        img *= 500.0
        return img

    def synthetic(self, size = (32, 32)):
        img = np.zeros(shape=size, dtype=np.float64)
        img[16:17, 16:17] = 30.0
        return img

    def load_video(self, path, num_repeats=8):
        vidcap = cv2.VideoCapture(path)
        success, image = vidcap.read()
        count = 0
        frames = []
        while success:
            img = image
            img = img[:872, :872, :]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
            img = cv2.resize(img, (32, 32))
            img /= 255.0
            img *= 500.0
            f = []
            for i in range(num_repeats):
                f.append(img)
            frames.append(np.array(f))
            #cv2.imwrite("frame%d.jpg" % count, img)     # save frame as JPEG file      
            del img
            success,image = vidcap.read()
            print('Read a new frame: ', success)
            count += 1
            #if count == 2:
            #    break
        frames = [x for i, x in enumerate(frames) if i % 10 == 0]
        return frames

class Network:
    def __init__(self, num_neurons:int, encodings:np.ndarray, dt:float = 0.1, shape = (28, 28)) -> None:
        self.shape = shape
        # Initialize weight matrix
        self.num_neurons = num_neurons + encodings.shape[0]
        self.weights = WeightMatrix(self.num_neurons, 0.001, 0.002)
        self.encodings = encodings
        # Adjust weights between encodings and hidden
        self.weights.weights[-self.encodings.shape[0]:] = 0.5

        self.neurons = dict()

        v_rest = np.random.uniform(-66.0, -64.0, self.num_neurons).astype(np.float64)
        v_reset = v_rest - np.float64(5.0)
        tau = np.random.uniform(19.5, 20.5, self.num_neurons).astype(np.float64)
        self.thresh = np.random.uniform(-55.0, -45.0, self.num_neurons).astype(np.float64)
        self.init_potentials = v_rest + np.float64(2.71)
        for i in range(self.num_neurons):
            if i >= self.num_neurons - self.encodings.shape[0]:
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
        #structured_array = np.empty(self.num_neurons, dtype=dtype)
        array = np.empty((self.num_neurons, len(keys)), dtype=np.float64)
        for i in self.neurons:
            neuron = self.neurons[i]
            array[i] = tuple(neuron[key] for key in keys)

        # Access as structured array fields
        self.neurons = array
        del array
        del keys

        self.init_spikes = np.zeros(shape=(self.num_neurons), dtype=np.float64)
        self.init_step = True
        self.post_spikes = self.init_spikes
        self.post_tau = tau
        self.pre_tau = tau
        self.pre_spikes = self.init_spikes
        self.spike_readout = []

        self.global_step_tick = 0
        self.clip_interval = 8
        self.error_thresholds = np.linspace(1.0, 0.1, num=10)

    def step(self, input_signals):
        for ni in range(self.num_neurons):
            if ni < self.num_neurons - self.encodings.shape[0]:
                wp, new_p = LIF.step(self.neurons[ni], input_signals[ni])
                self.neurons[ni][0] = new_p
                self.post_spikes[ni] = wp
            elif ni >= self.num_neurons - self.encodings.shape[0]:
                self.post_spikes[ni] = self.thresh[ni]

        self.post_spikes = self.weights.compute_spikes(self.post_spikes, self.thresh)
        self.pre_spikes = self.weights.compute_spikes(self.pre_spikes, self.thresh)
        #print(input_signals[0:self.num_neurons - self.encodings.shape[0]])
        #raise
        # Normalize the input signal and output signals
        norm_input  = normalize(input_signals[:self.num_neurons - self.encodings.shape[0]])
        norm_output = normalize(self.post_spikes[:self.num_neurons - self.encodings.shape[0]])
        # Compute the difference between the 2 norms as an error vector
        error_vector = np.abs(norm_input - norm_output)
        # Test:
        #   1. Shut off neurons above 1.0
        #   2. Shut down weights between neurons above 1.0
        #   3. Reduce weights between neurons above 1.0 to the bare minimum (1e-4)
        #   4. Additionally step neurons above 1.0?
        #print(error_indices)
        # Test case 1.
        
        # Get neurons where error is above error threshold
        for th in self.error_thresholds:
            error_indices = np.where(error_vector > np.float64(th))
            if len(error_indices) > 0:
                for idx in error_indices:
                    self.post_spikes[idx] = self.neurons[idx, 4]
                break

        if self.global_step_tick % self.clip_interval:
            self.weights.update_weights_combined(
                self.pre_spikes, 
                self.post_spikes, 
                self.pre_tau, 
                self.post_tau, 
                clip=True, 
                top_k=self.num_neurons//4,
                error = error_vector
            )
        else:
            self.weights.update_weights_combined(
                self.pre_spikes, 
                self.post_spikes, 
                self.pre_tau, 
                self.post_tau, 
                clip=False, 
                top_k=self.num_neurons//4,
                error = error_vector
            )

        signals = self.weights.propagate_signals(self.post_spikes, method='mean')
        self.pre_spikes = self.post_spikes
        return signals

    def run(self, data_stream):
        total_epochs = len(data_stream)
        #with open('step_log.txt', 'w+') as step_debug_out:
        #    pass
        for xi, x in enumerate(data_stream):
            x: np.ndarray
            xi: int
            y = self.pre_spikes
            print(f'Epoch {xi+1}/{total_epochs}')
            for sub_x in tqdm(x):
                sub_x = sub_x.flatten()
                sub_x = np.append(sub_x, self.encodings[xi])
                sub_x = np.sum([sub_x, y], axis=0)
                y = self.step(sub_x)
            if xi % 4 == 0:
                self.weights.prune_weights(threshold=1e-4)
            #print(f"Weight changes from encoding neurons: {np.diff(self.weights.weights[-self.encodings.shape[0]:, :]).sum(axis=0)}")
            #self.neurons[:, 0] = self.init_potentials

    def recall(self, encoding, num_ticks=8):
        recall_input = np.zeros(shape=(self.num_neurons))
        recall_input[-self.encodings.shape[0]:] = encoding
        recall_spikes = np.zeros(shape=(self.num_neurons))
        #raise
        signals = recall_input
        for i in range(num_ticks):
            spike_cache = []
            if i > 0:
                recall_input = np.sum([recall_input, signals], axis=0)
            for ni in range(self.num_neurons):
                wp, new_p = LIF.step(self.neurons[ni], recall_input[ni])
                if np.isnan(wp):
                    print(i, ni)
                    print(wp.dtype)
                    raise
                spike_cache.append(wp)
            #print(spike_cache)
            spike_cache = np.array(spike_cache, dtype=np.float64)
            spike_cache = self.weights.compute_spikes(spike_cache, self.thresh)
            #print(spike_cache)
            signals = self.weights.propagate_signals(spike_cache, method='sum')
            recall_spikes = np.sum([spike_cache, recall_spikes], axis=0)

        return recall_spikes

    def infer(self, samples=[], encodings = [[0, 500], [500, 0]]):
        fig, ax = plt.subplots(len(encodings), 3)

        for i, enc in enumerate(encodings):
            recall_spikes = self.recall(enc, 255)
            recall_spikes = recall_spikes[:self.num_neurons-len(self.encodings)]
            readout = np.reshape(recall_spikes[:1024], newshape=(32, 32))
            readout = cv2.normalize(readout, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            
            groundtruth = cv2.normalize(samples[i], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

            diff = np.abs(samples[i] - readout)
            diff = cv2.normalize(diff, None, 0, 1, norm_type=cv2.NORM_MINMAX)
            error = np.square(diff).mean()
            ax[i, 0].imshow(groundtruth, cmap='gray')
            ax[i, 0].title.set_text('Groundtruth')
            ax[i, 1].imshow(readout, cmap='gray')
            ax[i, 1].title.set_text('Recall')
            ax[i, 2].imshow(diff, cmap='viridis')
            ax[i, 2].title.set_text(f'Diff: {error}')
        plt.show()
        fig.savefig('./results.png', dpi=300)

loader = DataLoader()
#img = loader.load_image('./BrainCoral.jpg')
frames = loader.load_video('./lichen.mp4', num_repeats=32)
# Generate encodings
encodings = np.fliplr(np.eye(len(frames), dtype=np.float64)*500.0)
num_neurons = 32 * 32
#raise
net = Network(num_neurons, encodings, dt=1.0)
#net.run(data_stream=[img for i in range(32)])
net.run(data_stream=frames)
net.infer([i[0] for i in frames], encodings)