import cProfile
import pstats
from main import Network, DataLoader
import numpy as np
import dataset
# Function to profile
def main():
    loader = DataLoader()
    #img = loader.load_image('./BrainCoral.jpg')
    frames = loader.load_video('./lichen.mp4', num_repeats=32)
    
    # Generate encodings
    encodings = np.fliplr(np.eye(len(frames), dtype=np.float64)*500)
    num_neurons = 32 * 32
    print(num_neurons)
    #raise
    net = Network(num_neurons, encodings, dt=0.5)
    #net.run(data_stream=[img for i in range(32)])
    net.run(data_stream=frames)
    net.infer([i[0] for i in frames], encodings)

# Profile the function
if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    
    # Save and print profiling stats
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('time')  # Sort by execution time
    stats.print_stats(20)     # Print the top 20 time-consuming functions
