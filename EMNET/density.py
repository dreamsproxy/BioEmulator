import numpy as np
from itertools import combinations
from collections import OrderedDict
from scipy.spatial.distance import pdist
import tqdm
from vispy import app, scene, visuals

def generate_cube(size, num_particles):
    total_points = size * size * size
    cube = np.zeros(shape=total_points, dtype=np.float64)
    indices = np.random.randint(low=0, high=total_points-1, size=num_particles)
    cube[indices] = 1.0
    cube = cube.reshape((size, size, size))
    positions = np.array(np.where(cube >= 1.0)).T

    node_dict = OrderedDict({i: tuple(c.tolist()) for i, c in enumerate(positions)})
    return cube, positions, node_dict

def get_all_distances(id, distance_dict):
    # Filter all distances involving the given ID
    result = {pair: dist for pair, dist in distance_dict.items() if id in pair}
    return result

if __name__ == '__main__':
    cube, coords, pos_dict = generate_cube(128, 256)  # Use smaller cube for visibility
    distances = pdist(coords, metric='euclidean')
    distances /= np.float64(cube.shape[0]-1)  # Normalize by cube size
    id_pairs = list(combinations(pos_dict.keys(), 2))
    distance_dict = {pair: dist for pair, dist in zip(id_pairs, distances)}

    top_k_samples = 3
    top_completed_pairs = []
    top_connections = []
    for k in tqdm.tqdm(pos_dict.keys()):
        all_distances = get_all_distances(k, distance_dict)
        sorted_pairs = sorted(all_distances.items(), key=lambda x: x[1])  # Sort by distance
        k_top_samples = [pair for pair, _ in sorted_pairs[:top_k_samples] if pair not in top_completed_pairs]
        
        top_completed_pairs.extend(k_top_samples)
        for pair in k_top_samples:
            other_id = pair[1] if pair[0] == k else pair[0]
            top_connections.append([coords[k], coords[other_id]])  # Add connection as a pair of points
    
    bot_k_samples = 2
    bot_completed_pairs = []
    bot_connections = []
    for k in tqdm.tqdm(pos_dict.keys()):
        all_distances = get_all_distances(k, distance_dict)
        sorted_pairs = sorted(all_distances.items(), key=lambda x: x[1], reverse=True)  # Sort by distance
        k_top_samples = [pair for pair, _ in sorted_pairs[:bot_k_samples] if pair not in bot_completed_pairs]
        
        bot_completed_pairs.extend(k_top_samples)
        for pair in k_top_samples:
            other_id = pair[1] if pair[0] == k else pair[0]
            bot_connections.append([coords[k], coords[other_id]])  # Add connection as a pair of points
    
    bot_connections = np.array(bot_connections).reshape(-1, 3)  # Reshape to (N, 3) for visualization
    bot_connections = (bot_connections - 255.5) / 255.5

    top_connections = np.array(top_connections).reshape(-1, 3)  # Reshape to (N, 3) for visualization
    top_connections = (top_connections - 255.5) / 255.5
    coords = (coords - 255.5) / 255.5
    
    canvas = scene.SceneCanvas(keys='interactive', size=(800, 800), bgcolor='black')
    view = canvas.central_widget.add_view()
    view.camera = 'arcball'  # Add interactive 3D rotation

    # Scatter points
    scatter = scene.visuals.Markers()
    scatter.set_data(coords, edge_color='white', face_color='white', size=5)
    view.add(scatter)

    # Draw lines
    top_lines = scene.visuals.Line(pos=top_connections, color='blue', connect='segments')
    view.add(top_lines)
    bot_lines = scene.visuals.Line(pos=bot_connections, color='red', connect='segments')
    view.add(bot_lines)

    canvas.show()
    app.run()
