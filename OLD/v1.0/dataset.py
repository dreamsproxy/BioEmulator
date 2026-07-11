import cv2
import numpy as np
from glob import glob
import os
from tqdm import tqdm

def load_mnist():
    train_paths = sorted(glob('./mnist/mnist_png/train/*/'))
    train_paths = [i for i in train_paths if '0' in i or '1' in i]
    dataset = []
    for i, cpath in enumerate(train_paths):
        cpath = cpath.replace('\\', '/')
        class_images = []
        img_paths = [i.replace('\\', '/') for i in glob(os.path.join(cpath, '*.png'))][:64]
        for p in tqdm(img_paths):
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE).astype(np.float64)
            #img /= 255.0
            #img *= 500.0
            class_images.append(img)
        dataset.append(np.array(class_images, dtype=np.float64))
    dataset = np.array(dataset, dtype=np.float64)
    encodings = np.fliplr(np.eye(len(dataset), dtype=np.float64) * 500)
    #dataset = batch(dataset, batch_size=32, labels = encodings)
    return dataset, encodings

def batch(dataset:np.ndarray, batch_size:int = 16, labels=None) -> np.ndarray:
    num_classes = dataset.shape[0]
    indices = [i for i in range(dataset.shape[1])]
    total_samples = dataset.shape[0] * dataset.shape[1]
    pairs = []
    for i in range(num_classes):
        for j in indices:
            pairs.append((dataset[i][j], labels[i]))
    #pairs = np.array(pairs, dtype=object)
    np.random.shuffle(pairs)
    pairs = np.array_split(pairs, total_samples//batch_size)
    for i, b in enumerate(pairs):
        pairs[i] = b.tolist()
        #print(pairs[i])
    return pairs