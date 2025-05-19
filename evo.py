import numpy as np
import cv2
from matplotlib import pyplot as plt
from numba import njit

def load_gif(path, size:tuple[int, int]=(256, 256)) -> np.ndarray:
    # capture the animated gif
    gif = cv2.VideoCapture(path)
    frames = []
    ret, frame = gif.read()  # ret=True if it finds a frame else False.
    while ret:
        # read next frame
        ret, frame = gif.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #h, w = frame.shape
        #x = w/2 - size[1]/2
        #y = h/2 - size[0]/2
        #frame = frame[int(y):int(y+size[0]), int(x):int(x+size[1])]
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        frame = frame.astype(np.float64)
        frame = (frame - 127.5) / 127.5
        frame = frame.reshape((size[0], size[1], 1))
        frame = cv2.resize(frame, (32, 32))
        frames.append(frame)
    frames = np.array([f for i, f in enumerate(frames) if i % 2 == 0])
    return frames, np.eye(len(frames), len(frames), dtype=np.float64)

@njit
def softmax_nonzero_columns(arr):
    rows, cols = arr.shape
    result = np.zeros_like(arr, dtype=np.float64)
    
    for j in range(cols):  # Iterate over columns
        # Get nonzero values in the column
        mask = arr[:, j] != 0
        if np.any(mask):
            col_nonzero = arr[mask, j]
            max_val = np.max(col_nonzero)  # Stability trick
            exp_vals = np.exp(col_nonzero - max_val)  # Subtract max to avoid overflow
            sum_exp = np.sum(exp_vals)
            
            # Compute softmax only for nonzero elements
            result[mask, j] = exp_vals / sum_exp
    
    return result

states = np.random.uniform(-1.0, 1.0, 64).astype(np.float64)
pool_weights = np.random.standard_normal((64, 64)).astype(np.float64)
input_weights = np.random.standard_normal((64, 1024)).astype(np.float64)
output_weights = np.random.standard_normal((64, 1024)).astype(np.float64)

data, labels = load_gif('3a737eff8efabd395c8a076a87cbac00.gif')
input_data = data[0].flatten()

# Do hidden steps
pool_outputs = np.dot(input_weights, input_data) + np.dot(pool_weights, states.reshape(-1, 1))
states = np.mean([states, pool_outputs[1]])

# Do hidden -> reconstruction steps
reco_outputs = np.dot(output_weights.T, pool_outputs) + np.dot(output_weights.T, pool_outputs)
reco_outputs = np.reshape(np.mean(reco_outputs.T, axis=0), (32, 32))
reco_outputs = np.tanh(reco_outputs)

diff_map = reco_outputs - data[0]
diff_map = np.expand_dims(diff_map,-1)
print(diff_map.shape)
error = np.mean(np.square(diff_map), axis=-1)
print(states.mean())
print(error.mean())
print(error.shape)