import numpy as np
import cv2
import numba
from matplotlib import pyplot as plt
from itertools import combinations
from glob import glob

def step_func(params:np.ndarray[np.float32], x:np.ndarray[np.float32]):
    state       = np.expand_dims(params[0], 0)
    multiplier  = np.expand_dims(params[1], 0)
    leak        = np.expand_dims(params[2], 0)
    y           = np.mean(state * (x * multiplier), axis=(0, 1)) * leak
    return y, y * (1*np.mean(np.tanh(state*x)))

N_NEURONS = 512
K = np.array([
    [-1,  0, -1],
    [ 0,  1,  0],
    [-1,  0, -1]], dtype=np.float16)
K = np.tile(np.expand_dims(K, -1), 3)
states = []
outputs = []
network = np.random.normal(0.0, 2.71, size=(N_NEURONS, 3, 3)).astype(np.float32)
network[:, :, -1] = np.abs(network[:, -1])
weights = np.random.normal(0.0, 2.71, size=(N_NEURONS, 1)).astype(np.float32)
for img_id, p in enumerate(glob('./lichen/inputs/**')):
    x = cv2.imread(p, cv2.IMREAD_COLOR).astype(np.float32) / 127.5 - 1
    c = []
    ks = 3
    for i in range(3, x.shape[0], 1):
        for j in range(3, x.shape[1], 1):
            c.append(np.mean(np.matmul(K, x[i-3:i, j-3:j, :]), axis=(0, 1), dtype=np.float32))
    c = np.array(c, dtype=np.float32)
    outputs_cache = np.zeros((4096, 3))
    state_cache = np.zeros((N_NEURONS, 3))
    idx = 0
    for i in range(4096):
        if idx > N_NEURONS-1:
            idx = 0
            network[:, 0] *= state_cache # (network[:, 0] + state_cache) / 2
            states.append(state_cache)
        s, y = step_func(network[idx], c)
        state_cache[idx] = s
        outputs_cache[i] = y
        idx += 1
    outputs_cache = np.tanh(outputs_cache)
    outputs_cache = outputs_cache.reshape((64, 64, 3))
    outputs_cache = (outputs_cache + 1) * 127.5
    outputs_cache = outputs_cache.astype(np.uint8)
    cv2.imwrite(f'{img_id:04d}.png', outputs_cache)

raise


x = i_weights * x
print(x.shape)
output = np.zeros(shape=(4096))

for i in range(4096):
    if i > 1023:
        _tick = 0
    params = network[_tick]
    state, y = step_func(params, x[i])
    network[_tick][0] = state
    output[i] = y
    _tick += 1
