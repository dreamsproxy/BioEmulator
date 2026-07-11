import numpy as np
from numba import jit

class array_ops:
    def __init__(self):
        pass

    @staticmethod
    @jit(nopython=True)
    def minmax_scale(
        matrix: np.ndarray, 
        target_min: np.float64 = np.float64(-1.0),
        target_max: np.float64 = np.float64(1.0),
        axis: int=-1) -> np.ndarray:
        """
        Scales a 2D numpy array (weight matrix) to a target range using min-max scaling.
        The scaling is applied globally across the entire matrix, not per feature/row.
        
        Parameters:
        -----------
        matrix : np.ndarray
            2D input array to be scaled
        target_min : np.float64
            Minimum value of the target range
        target_max : np.float64
            Maximum value of the target range
            
        Returns:
        --------
        np.ndarray
            Scaled 2D array with values in [target_min, target_max]
            
        Notes:
        ------
        The formula used is:
            scaled = (x - min) * (target_max - target_min) / (max - min) + target_min
        """
        if axis == -1:
            # Handle edge case where all values are the same
            matrix_min = np.min(matrix)
            matrix_max = np.max(matrix)
            
            if np.abs(matrix_max - matrix_min) < np.finfo(np.float64).eps:
                # If all values are the same, return matrix filled with mean of target range
                return np.full_like(matrix, (target_max + target_min) / 2)
            
            # Compute scaling
            scale = (target_max - target_min) / (matrix_max - matrix_min)
            
            # Scale the matrix
            scaled_matrix = (matrix - matrix_min) * scale + target_min
            
            # Handle numerical precision issues
            scaled_matrix = np.minimum(scaled_matrix, target_max)
            scaled_matrix = np.maximum(scaled_matrix, target_min)
            
            return scaled_matrix
        else:
            if axis == 1:
                scaled_matrix = np.empty_like(matrix)
                for col in range(matrix.shape[axis]):
                    col_min = np.min(matrix[:, col])
                    col_max = np.max(matrix[:, col])
                    if np.abs(col_max - col_min) < np.finfo(np.float64).eps:
                        # If all values are the same, return matrix filled with mean of target range
                        scaled_matrix[:, col] =  np.full_like(matrix[:, col], (target_max + target_min) / 2)
                    col_scale = (target_max - target_min) / (col_max - col_min)
                    scaled_matrix[:, col] = (matrix[:, col] - col_min) * col_scale + target_min
                # Handle numerical precision issues
                scaled_matrix = np.minimum(scaled_matrix, target_max)
                scaled_matrix = np.maximum(scaled_matrix, target_min)
                return scaled_matrix

    @staticmethod
    def img2stream(img:np.ndarray, axis:int = 0):
        if axis == 0:
            return img
        elif axis == 1:
            return img.T
        else:
            raise NotImplementedError('Currently only supports col or row wise as sequence!')

    @staticmethod
    def kernelize(mat: np.ndarray, 
                kernel_size: tuple[int, int] = (7, 7), 
                strides: tuple[int, int] = (1, 1),
                stats: tuple[str] = ('mean', 'variance')) -> np.ndarray:
        """
        Extended kernelize to support multiple statistical features like variance.
        """
        result = []
        for yi in range(kernel_size[1], mat.shape[1], strides[1]):
            row_stats = []
            for xi in range(kernel_size[0], mat.shape[0], strides[0]):
                kernel = mat[yi - kernel_size[1]:yi, xi - kernel_size[0]:xi]
                stats_features = []
                if 'mean' in stats:
                    stats_features.append(np.mean(kernel))
                if 'variance' in stats:
                    stats_features.append(np.var(kernel))
                if 'max' in stats:
                    stats_features.append(np.max(kernel))
                row_stats.append(stats_features)
            result.append(row_stats)
        return np.array(result, dtype=np.float64).reshape(-1, len(stats))

import cv2
class dataloader:
    def __init__(self):
        pass

    @staticmethod
    def load_image(path, size:tuple[int, int] = (256, 256)):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if size != (0, 0):
            img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
        img = array_ops.kernelize(img, kernel_size=(7, 7), strides=(2, 2))
        img = array_ops.kernelize(img, kernel_size=(5, 5), strides=(2, 2))
        #img = array_ops.kernelize(img, kernel_size=(3, 3), strides=(2, 2))
        return img
    
    @staticmethod
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
            #center = frame.shape
            #x = center[1]/2 - size[1]/2
            #y = center[0]/2 - size[0]/2

            #frame = frame[int(y):int(y+size[0]), int(x):int(x+size[1])]
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            frame = frame.astype(np.float64)
            frame = (frame - 127.5) / 127.5
            frame = frame.reshape((size[0], size[1], 1))
            frames.append(frame)
        frames = np.array([f for i, f in enumerate(frames) if i % 2 == 0])
        return frames, np.eye(len(frames), len(frames), dtype=np.float64)

    @staticmethod
    def load_video(path, num_repeats=1):
        print('Loading video...')
        vidcap = cv2.VideoCapture(path)
        success, image = vidcap.read()
        count = 0
        frames = []
        while success:
            img = image
            h, w = img.shape[:2]
            if h > w:
                c = w
            elif w > h:
                c = h
            elif w == h:
                c = h
            img = img[:c, :c, :]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
            img = cv2.resize(img, (8, 8))
            #img = (img - 127.5) / 127.5
            img = (img - 127.5) / 127.5
            img *= 2.71
            frames.append(np.array(img))
            #cv2.imwrite("frame%d.jpg" % count, img)     # save frame as JPEG file      
            del img
            success,image = vidcap.read()
            print(f'Frame: {count}: {success}')
            count += 1
        new_frames = [frames[i] for i in range(len(frames)) if i % 5 == 0]
        new_frames = np.array(new_frames).reshape((len(new_frames), 8, 8, 1))
        encodings = np.eye(len(new_frames), len(new_frames), dtype=np.float64)
        print(f'Loaded {len(new_frames)} frames!')

        return [new_frames, encodings]