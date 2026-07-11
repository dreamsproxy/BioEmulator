import sounddevice as sd
import LIF
from WeightMatrix import WeightMatrix
import numpy as np
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm
from numba import njit
from scipy.io import wavfile
from scipy.signal import resample_poly
from multiprocessing import Pool, cpu_count

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
    
    def load_audio(self, path) -> np.ndarray:
        sr, data = wavfile.read(path)
        data = data.astype(np.float64)

        # Ensure 2 channels
        if data.ndim == 1:
            data = np.stack([data, data], axis=-1)
        elif data.shape[1] != 2:
            raise ValueError("Audio must have exactly 2 channels.")

        # Normalize to ~[0, 500] range
        data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 500.0
        return data  # shape: (samples, 2)
    
    def load_mono_audio(self, path, target_rate=22050):
        sr, data = wavfile.read(path)
        data = data.astype(np.float64)

        # Stereo to mono
        if data.ndim == 2:
            data = data.mean(axis=1)
        elif data.ndim == 1:
            pass  # Already mono
        else:
            raise ValueError("Unsupported audio shape")

        # Resample to target rate
        gcd = np.gcd(sr, target_rate)
        up = target_rate // gcd
        down = sr // gcd
        data = resample_poly(data, up, down)

        # Normalize to [0, 500]
        data -= np.min(data)
        data /= np.max(data)
        data *= 500.0

        return data  # shape: (samples,)

def _process_sample(args):
    i, stereo_sample, y_prev, num_neurons, thresh, weights, step_func = args
    # Prepare input vector
    sub_x = np.zeros(num_neurons, dtype=np.float64)
    sub_x[-2:] = stereo_sample  # last 2 neurons = [L, R] audio
    sub_x += y_prev

    # Step simulation
    y_new = step_func(sub_x)
    return (i, y_new)

class Network:
    def __init__(self, num_neurons:int, encodings:np.ndarray, dt:float = 0.1, output_shape = (28, 28)) -> None:
        self.output_shape = output_shape
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

    def step(self, input_signals, train=True):
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
        if train:
            # Normalize the input signal and output signals
            norm_input  = normalize(input_signals[:self.num_neurons - self.encodings.shape[0]])
            norm_output = normalize(self.post_spikes[:self.num_neurons - self.encodings.shape[0]])
            # Compute the difference between the 2 norms as an error vector
            error_vector = np.abs(norm_input - norm_output)

            # Get neurons where error is above error threshold
            for th in self.error_thresholds:
                error_indices = np.where(error_vector > np.float64(th))
                if len(error_indices) > 0:
                    for idx in error_indices:
                        self.post_spikes[idx] = self.neurons[idx, 4]
                    break

            if self.global_step_tick % self.clip_interval == 0:
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
            sub_x[-2:] = stereo_sample  # last 2 neurons = [L, R] audio
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

    def run_MP(self, data_stream: np.ndarray, record_output=False):
        """
        Run continuous training on raw stereo audio (shape: [n_samples, 2]) using multiprocessing.
        """
        outputs = [] if record_output else None
        print(f"Running on {len(data_stream)} audio frames with multiprocessing pool(4)...")

        y = self.pre_spikes

        with Pool(processes=4) as pool:
            tasks = [
                (i, stereo_sample, y, self.num_neurons, self.thresh, self.weights, self.step)
                for i, stereo_sample in enumerate(data_stream)
            ]
            for i, y_new in tqdm(pool.imap(_process_sample, tasks), total=len(tasks)):
                y = y_new  # carry forward most recent
                if i % 2048 == 0:
                    self.weights.prune_weights(threshold=1e-4)
                if record_output:
                    outputs.append(y[-2:].copy())

        if record_output:
            return np.array(outputs)

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

    def infer_image(self, samples=[], encodings = [[0, 500], [500, 0]]):
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

    def save_weights(self,path:str='weights.npy'):
        np.save(path, self.weights.weights)


def run_live_audio(net, input_rate=44100, target_rate=22050, blocksize=512):
    """
    Runs the network in real time on microphone input.
    - Downsamples from input_rate → target_rate
    - Upsamples output back to input_rate for playback
    """
    print('init...')

    # Buffer states
    net.pre_spikes = np.zeros_like(net.pre_spikes)
    net.post_spikes = np.zeros_like(net.post_spikes)
    y_prev = np.zeros(net.num_neurons, dtype=np.float64)
    def audio_callback(indata, outdata, frames, time, status):
        if status:
            print(status)

        # Convert mono input safely
        if indata.ndim == 2:
            x = indata[:, 0].astype(np.float64)  # take the first channel
        else:
            x = indata.astype(np.float64)

        # Convert to mono
        x = indata.mean(axis=1).astype(np.float64)

        # Downsample to target rate
        gcd = np.gcd(input_rate, target_rate)
        up = target_rate // gcd
        down = input_rate // gcd
        x_down = resample_poly(x, up=1, down=down)

        # Process as one block (vectorized)
        n = len(x_down)
        out_block = np.zeros((n, 2), dtype=np.float64)
        sub_x = np.zeros((n, net.num_neurons), dtype=np.float64)

        # Fill stereo channels
        sub_x[:, -2:] = np.stack([x_down, x_down], axis=-1)

        # Run sequentially but faster: less Python overhead
        y_local = np.copy(y_prev)
        for i in range(n):
            y_local = net.step(sub_x[i], train=False)
            out_block[i] = y_local[-2:]

        # Update previous spikes
        y_prev[:] = y_local

        # Upsample back to playback rate
        out_up = resample_poly(out_block.mean(axis=1), up=down, down=1)
        out_up = np.clip(out_up, -1.0, 1.0)

        # Stereo output
        out_up = np.repeat(out_up[:, np.newaxis], 2, axis=1)

        # Ensure we don’t overflow buffer
        if len(out_up) < len(outdata):
            outdata[:len(out_up), :] = out_up
            outdata[len(out_up):, :] = 0
        else:
            outdata[:] = out_up[:len(outdata)]

    input_device_index = 9   # your mic
    output_device_index = 14  # your speakers
    with sd.Stream(
        samplerate=input_rate,
        blocksize=1024,
        dtype='float32',
        channels=(1, 2),  # 👈 1 input, 2 output
        callback=audio_callback,
        device=(input_device_index, output_device_index),
        latency='high'
    ):

        print(f"🎤 Using input #{input_device_index}, output #{output_device_index}")
        while True:
            sd.sleep(1000)
if __name__ == "__main__":
    SAMPLE_RATE = 48000
    #print(sd.query_devices())
    #print("\nDefault input:", sd.default.device[0])
    #print("Default output:", sd.default.device[1])
    #raise

    loader = DataLoader(
        
    )
    net = Network(num_neurons=32, encodings=np.zeros((1,)), dt=0.5)

    # Live test
    run_live_audio(net, input_rate=48000, target_rate=SAMPLE_RATE)

#if __name__ == "__main__":
#    SAMPLE_RATE = 22010
#    loader = DataLoader()
#    #img = loader.load_image('./BrainCoral.jpg')
#    #frames = loader.load_video('./lichen.mp4', num_repeats=32)
#    #audio_seq = loader.load_audio('./SLOW DANCING IN THE DARK.wav')
#    encodings = loader.load_mono_audio('./SLOW DANCING IN THE DARK.wav', SAMPLE_RATE)  # shape (n_samples, 2)
#    # Trim to first 30 seconds
#    encodings = encodings[:int(SAMPLE_RATE*5)]
#    # Define network: 32 LIF + 2 input channels
#    net = Network(num_neurons=64, encodings=np.zeros((1,)), dt=0.5)
#    #net.weights.weights = np.load('./weights.npy')
#    net.run_MP(encodings)
#    output_audio = []
#    validation_sample = encodings[:SAMPLE_RATE*5]
#
#    #for i in range(0, len(validation_sample)):
#    #    if i % 2 == 0:
#    #        validation_sample[i] = 0.0
#
#    # Reset state for inference
#    net.pre_spikes = np.zeros_like(net.pre_spikes)
#    net.post_spikes = np.zeros_like(net.post_spikes)
#    net.global_step_tick = 0
#
#    output_audio = []
#    test_audio = loader.load_mono_audio('Kudi.wav', SAMPLE_RATE)  # shape (n_samples, 2)
#    # Trim to first 30 seconds
#    test_audio = test_audio[:int(SAMPLE_RATE*5)]
#
#    for frame in tqdm(test_audio):
#        sub_x = np.zeros(net.num_neurons, dtype=np.float64)
#        sub_x[-2:] = frame  # feed stereo sample
#        y = net.step(sub_x, train=False)
#        output_audio.append(y[-2:].copy())
#
#    output_audio = np.array(output_audio)
#
#    # Normalize to [-1, 1] then scale to int16
#    output_audio -= output_audio.mean(axis=0)  # zero-center
#    output_audio /= np.max(np.abs(output_audio)) + 1e-9  # normalize
#    output_audio *= 32767  # scale to int16
#    output_audio = output_audio.astype(np.int16)
#
#    # Save as WAV
#    wavfile.write("model_output.wav", SAMPLE_RATE, output_audio)
#    net.save_weights()