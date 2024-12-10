import LIF
from WeightMatrix import WeightMatrix
import numpy as np
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

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
            if count == 3:
                break
        return frames

class Network:
    def __init__(self, num_neurons:int, encodings:np.ndarray, dt:float = 0.1) -> None:
        # Initialize weight matrix
        self.num_neurons = num_neurons + len(encodings)
        self.weights = WeightMatrix(self.num_neurons, 0.01, 0.012)
        self.encodings = encodings
        # Adjust weights between encodings and hidden
        self.weights.weights[self.num_neurons - len(self.encodings):] = 1.0

        self.neurons = dict()

        v_rest = np.random.uniform(-66.0, -64.0, self.num_neurons)
        v_reset = v_rest - 5.0
        tau = np.random.uniform(19.5, 20.5, self.num_neurons)
        self.thresh = np.random.uniform(-55.0, -50.0, self.num_neurons)

        for i in range(self.num_neurons):
            self.neurons[i] = {
                'potential': v_rest[i],
                'dt': dt,
                'tau': tau[i],          # Membrane time constant (ms)
                'v_rest': v_rest[i],      # Resting potential (mV)
                'v_reset': v_reset[i],     # Reset potential after spike (mV)
                'v_threshold': self.thresh[i]  # Firing threshold (mV)
            }

        self.init_spikes = np.zeros(shape=(self.num_neurons))
        self.init_step = True
        self.post_spikes = self.init_spikes
        self.post_tau = tau
        self.pre_tau = tau
        self.pre_spikes = self.init_spikes
        self.spike_readout = []

        self.position_matrix = np.random.uniform(0.0, 1.0, (self.num_neurons, 3))
        self.distance_matrix = self.compute_distance_matrix(self.position_matrix)
        print(self.distance_matrix.shape)
        raise
    
    def compute_distance_matrix(self, points):
        """
        Computes the pairwise Euclidean distance matrix for an array of points.

        Parameters:
            points (np.ndarray): Input array of shape (N, 3), where N is the number of points.

        Returns:
            np.ndarray: Distance matrix of shape (N, N).
        """
        # Validate the input array
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Input array must have shape (N, 3)")
        
        # Compute the pairwise Euclidean distance matrix
        diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diffs ** 2, axis=-1))
        
        return distances

    def step(self, input_signals):
        # Temporary cache of spikes
        for ni in range(self.num_neurons):
            #print(self.neurons.keys())
            assert self.num_neurons == len(input_signals)
            if ni < self.num_neurons-len(self.encodings):
                wp, new_p = LIF.step(LIF.parse_params(self.neurons[ni]), input_signals[ni])
                self.neurons[ni]['potential'] = new_p
                self.post_spikes[ni] = wp
            elif ni>= self.num_neurons-len(self.encodings):
                self.post_spikes[ni] = self.thresh[ni]

        self.post_spikes = self.weights.compute_spikes(self.post_spikes, self.thresh)
        self.pre_spikes = self.weights.compute_spikes(self.pre_spikes, self.thresh)
        self.weights.update_weights_combined(self.pre_spikes, self.post_spikes, self.pre_tau, self.post_tau)

        # Propagate signals
        signals = self.weights.propagate_signals(self.post_spikes, method='mean')
        self.weights.prune_weights(threshold=1e-4)
        self.pre_spikes = self.post_spikes
        return signals

    def recall(self, encoding, num_ticks=8):
        recall_input = np.ones(shape=(self.num_neurons))
        recall_input[self.num_neurons-len(self.encodings):] = encoding
        recall_spikes = np.zeros(shape=(self.num_neurons))

        signals = recall_input
        for i in range(num_ticks):
            spike_cache = []
            if i > 0:
                recall_input = np.sum([recall_input, signals], axis=0)
            for ni in range(self.num_neurons):
                wp, new_p = LIF.step(LIF.parse_params(self.neurons[ni]), recall_input[ni])
                #print(wp)
                spike_cache.append(wp)
            spike_cache = self.weights.compute_spikes(spike_cache, self.thresh)
            signals = self.weights.propagate_signals(spike_cache, method='sum')
            recall_spikes = np.sum([spike_cache, recall_spikes], axis=0)

        return recall_spikes

    def run(self, data_stream):
        for xi, x in enumerate(data_stream):
            x: np.ndarray
            y = self.pre_spikes
            for sub_x in tqdm(x):
                sub_x = sub_x.flatten()
                sub_x = np.append(sub_x, self.encodings[xi, :])
                sub_x = np.sum([sub_x, y], axis=0)
                y = self.step(sub_x)

    def infer(self, samples=[], encodings = [[0, 500], [500, 0]]):
        #print(len(samples), len(encodings))
        assert len(samples) == len(encodings)

        fig, ax = plt.subplots(len(encodings), 3)

        for i, enc in enumerate(encodings):
            recall_spikes = self.recall(enc, 255)
            recall_spikes = recall_spikes[:self.num_neurons-len(self.encodings)]
            readout = np.reshape(recall_spikes[:1024], newshape=(32, 32))
            readout = cv2.normalize(readout, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            
            groundtruth = cv2.normalize(samples[i], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

            diff = np.abs(samples[i] - readout)
            diff = cv2.normalize(diff, None, 0, 1, norm_type=cv2.NORM_MINMAX)
            ax[i, 0].imshow(groundtruth, cmap='gray')
            ax[i, 0].title.set_text('Groundtruth')
            ax[i, 1].imshow(readout, cmap='gray')
            ax[i, 1].title.set_text('Recall')
            ax[i, 2].imshow(diff, cmap='viridis')
            ax[i, 2].title.set_text('Diff')
        plt.show()
        plt.savefig('./results.png', dpi=300)
        #return groundtruth, readout, diff

loader = DataLoader()
#img = loader.load_image('./BrainCoral.jpg')
frames = loader.load_video('./lichen.mp4', num_repeats=32)

# Generate encodings
encodings = np.fliplr(np.eye(len(frames), dtype=np.float64)*500)
num_neurons = 32 * 32
print(num_neurons)
#raise
net = Network(num_neurons, encodings, dt=1.0)
#net.run(data_stream=[img for i in range(32)])
net.run(data_stream=frames)
net.infer([i[0] for i in frames], encodings)