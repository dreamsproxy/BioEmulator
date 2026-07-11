import LIF
from WeightMatrix import WeightMatrix
import numpy as np
import cv2
from tqdm import tqdm
from numba import njit
from scipy.io import wavfile
from scipy.signal import resample_poly
from LIF import batch_lif_step

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
    
    def load_audio(self, path, target_rate=22050) -> np.ndarray:
        sr, data = wavfile.read(path)
        data = data.astype(np.float64)

        # Ensure 2 channels
        if data.ndim == 1:
            data = np.stack([data, data], axis=-1)
        elif data.shape[1] != 2:
            raise ValueError("Audio must have exactly 2 channels.")
        # Resample
        gcd = np.gcd(sr, target_rate)
        up = target_rate // gcd
        down = sr // gcd
        data = resample_poly(data, up, down)

        # Now write a diagnostic file *before* further scaling,
        # as 32-bit floats in -1.0..1.0 range:
        # (this makes sure it's valid WAV you can listen to)
        norm = data / (np.max(np.abs(data)) + 1e-9)
        wavfile.write("model_input.wav", target_rate, norm.astype(np.float32))
        # Normalize to ~[0, 500] range
        data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 500.0
        return data  # shape: (samples, 2)
    
    def load_mono_audio(self, path, target_rate=22050):
        sr, data = wavfile.read(path)
        data = data.astype(np.float64)

        # Stereo → mono
        if data.ndim == 2:
            data = data.mean(axis=1)
        elif data.ndim != 1:
            raise ValueError("Unsupported audio shape")

        # Resample
        gcd = np.gcd(sr, target_rate)
        up = target_rate // gcd
        down = sr // gcd
        data = resample_poly(data, up, down)

        # Now write a diagnostic file *before* further scaling,
        # as 32-bit floats in -1.0..1.0 range:
        # (this makes sure it's valid WAV you can listen to)
        norm = data / (np.max(np.abs(data)) + 1e-9)
        wavfile.write("model_input.wav", target_rate, norm.astype(np.float32))

        # Then scale to [0,500] for your model
        data -= data.min()
        data /= data.max() + 1e-9
        data *= 500.0

        return data  # (samples,)

class Network:
    def __init__(self, num_neurons:int, encodings:np.ndarray, dt:float = 0.1, output_shape = (28, 28)) -> None:
        self.output_shape = output_shape
        # Initialize weight matrix
        self.num_hidden = num_neurons
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

    def step(self, input_signals:np.ndarray, train:bool=True):
        #for ni in range(self.num_neurons):
        #    if ni < self.num_neurons - self.encodings.shape[0]:
        #        wp, new_p = LIF.step(self.neurons[ni], input_signals[ni])
        #        self.neurons[ni][0] = new_p
        #        self.post_spikes[ni] = wp
        #    elif ni >= self.num_neurons - self.encodings.shape[0]:
        #        self.post_spikes[ni] = self.thresh[ni]
        
        hidden_n = self.num_hidden
        # Update hidden neurons with the batched LIF update:
        batch_lif_step(
            self.neurons[:hidden_n],
            input_signals[:hidden_n],
            self.thresh[:hidden_n],
            self.post_spikes[:hidden_n]
        )
        # For encoding neurons, set the spike value directly (as before)
        self.post_spikes[-self.encodings.shape[0]:] = self.thresh[-self.encodings.shape[0]:]
        # Compute STDP Diffs
        self.post_spikes = self.weights.compute_spikes(self.post_spikes, self.thresh)
        self.pre_spikes = self.weights.compute_spikes(self.pre_spikes, self.thresh)
        # Compute the error vector (used for weight updates)
        norm_input  = normalize(input_signals[:self.num_neurons - self.encodings.shape[0]])
        norm_output = normalize(self.post_spikes[:self.num_neurons - self.encodings.shape[0]])
        #print(input_signals.shape)
        #error_vector = np.abs(norm_input[-N_CHANNELS:] - input_signals[-N_CHANNELS:])
        error_vector = np.abs(norm_input - norm_output)

        if train:
            # Suppress neurons with high error
            for th in self.error_thresholds:
                error_indices = np.where(error_vector > np.float64(th))
                if len(error_indices) > 0:
                    for idx in error_indices:
                        self.post_spikes[idx] = self.neurons[idx, 4]
                    break

            # Weight update step
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
            self.global_step_tick += 1

        # Signal propagation always happens
        signals = self.weights.propagate_signals(self.post_spikes, method='mean')
        self.pre_spikes = self.post_spikes
        return signals

    def run(self, data_stream: np.ndarray, record_output=False):
        """
        Run continuous training on raw stereo audio (shape: [n_samples, 2]).
        
        Args:
            data_stream: ndarray of shape (N, 2) representing stereo audio.
            record_output: If True, returns list of [L_out, R_out] per step.
        """
        outputs = [] if record_output else None
        print(f"Running on {len(data_stream)} audio frames...")

        y = self.pre_spikes
        for i, stereo_sample in enumerate(tqdm(data_stream)):
            # Prepare input vector
            sub_x = np.zeros(self.num_neurons, dtype=np.float64)
            sub_x[-N_CHANNELS:] = stereo_sample  # last 2 neurons = [L, R] audio
            sub_x += y  # add previous output spikes

            # Step simulation
            y = self.step(sub_x)

            # Prune every N steps (optional)
            if i % 2048 == 0:
                self.weights.prune_weights(threshold=1e-4)

            # Capture output if needed
            if record_output:
                outputs.append(y[-2:].copy())  # [L_out, R_out]

        if record_output:
            return np.array(outputs)

    def save_weights(self,path:str='weights.npy'):
        np.save(path, self.weights.weights)

SAMPLE_RATE = 22010
N_CHANNELS = 2
loader = DataLoader()
stream = loader.load_audio('./SLOW DANCING IN THE DARK.wav', SAMPLE_RATE)  # shape (n_samples, 2)
# Trim to first 30 seconds
stream = stream[:int(SAMPLE_RATE*5)]
# Define network: 32 LIF + 2 input channels
net = Network(num_neurons=64, encodings=np.zeros((N_CHANNELS,)), dt=1.0)
net.run(stream)
#net.weights.weights = np.load('./weights.npy')
net.save_weights()

validation_sample = stream[:SAMPLE_RATE*5]
output_audio = []

#for i in range(0, len(validation_sample)):
#    if i % 2 == 0:
#        validation_sample[i] = 0.0

for xi, sample in enumerate(tqdm(stream[:int(SAMPLE_RATE*5)])):  # just a short preview
    sub_x = np.zeros(net.num_neurons, dtype=np.float64)
    sub_x[-N_CHANNELS:] = sample
    sub_x += net.pre_spikes
    y = net.step(sub_x, train=False)
    output_audio.append(y[-N_CHANNELS:].copy())

output_audio = np.array(output_audio)
# Normalize to [-1, 1] then scale to int16
output_audio -= output_audio.mean()  # zero-center
output_audio /= np.max(np.abs(output_audio)) + 1e-9  # normalize
output_audio *= 32767  # scale to int16
output_audio = output_audio.astype(np.int16)
print(output_audio.shape)
print(output_audio.mean(), output_audio.std())
#raise

# Save as WAV
wavfile.write("model_output.wav", SAMPLE_RATE, output_audio)
## Normalize into -1.0..1.0
#out = np.array(output_audio, dtype=np.float64)
#out -= out.mean(axis=0)
#out /= np.max(np.abs(out)) + 1e-9
#
## Convert to int16 PCM
#pcm = (out * 32767).astype(np.int16)
#
## Save with correct sample-rate
#wavfile.write("model_output.wav", SAMPLE_RATE, pcm)