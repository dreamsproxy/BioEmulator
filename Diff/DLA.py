import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm, trange
import random

# Define the 3D grid size
GRID_SIZE = 50
CENTER = GRID_SIZE // 2
DIVERGENCE_THRESHOLD = 5
GLOBAL_AVG_DENSITY = 0
DENSITY_LOG_CACHE = []

# Initialize the grid with a single seed particle at the center
grid = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=bool)
grid[CENTER, CENTER, 49] = True

# Directions for random walk in 3D (6 possible moves)
directions = [
    (1, 0, 1),
    (0, 1, 1),
    (-1, 0, 1),
    (0, -1, 1),
    
    (1, 1, 1),
    (1, -1, 1),

    (-1, 1, 1),
    (-1, -1, 1),

    (-1, 0, 0), (1, 0, 0),
    (0, -1, 0), (0, 1, 0),
    (1, 1, 0), (-1, -1, 0),
    (1, -1, 0), (-1, 1, 0)
]

no_z = [
    (-1, 0, 0), (1, 0, 0),
    (0, -1, 0), (0, 1, 0),
    (1, 1, 0), (-1, -1, 0),
    (1, -1, 0), (-1, 1, 0)
]

def count_neighbors(x, y, z):
    # Calculate the density of neighbors within the given radius
    subgrid = grid[max(0, x-2):min(GRID_SIZE, x+2),
                   max(0, y-2):min(GRID_SIZE, y+2),
                   max(0, z-2):min(GRID_SIZE, z+2)]
    # Count the number of neighbors
    count = np.sum(subgrid)
    if count > 3:
        print("Density Warn!")
    DENSITY_LOG_CACHE.append(count)
    return count

# Define a function to perform the random walk
def random_walk(timestep):
    while True:
        x, y, z = [random.randint(0, GRID_SIZE-1) for _ in range(3)]
        n_neighbors = count_neighbors(x, y, z)
        while True:
            if any(
                (0 <= x+dx < GRID_SIZE) and (0 <= y+dy < GRID_SIZE) and (0 <= z+dz < GRID_SIZE) and grid[x+dx, y+dy, z+dz]
                for dx, dy, dz in directions
            ) and n_neighbors <= 2:
                return x, y, z
            elif n_neighbors > 2:
                dx, dy, dz = (0, 0, -1)
                x, y, z = (x + dx) % GRID_SIZE, (y + dy) % GRID_SIZE, (z + dz) % GRID_SIZE
                return x, y, z
            dx, dy, dz = random.choice(directions)
            if z + dz >= GRID_SIZE:
                dx, dy, dz = random.choice(no_z)
            x, y, z = (x + dx) % GRID_SIZE, (y + dy) % GRID_SIZE, (z + dz) % GRID_SIZE

# Simulate the DLA process
def simulate_dla(steps):
    positions = [(CENTER, CENTER, 50)]
    for _ in trange(steps):
        x, y, z = random_walk(_)
        grid[x, y, z] = True
        positions.append((x, y, z))
    GLOBAL_AVG_DENSITY = np.mean(DENSITY_LOG_CACHE)
    print(GLOBAL_AVG_DENSITY)
    return positions

# Animate the DLA process using Plotly
def animate_dla(positions):
    frames = []
    for i in range(len(positions)):
        x_data, y_data, z_data = zip(*positions[:i+1])
        frame = go.Frame(data=[go.Scatter3d(
            x=x_data, y=y_data, z=z_data,
            mode='markers',
            marker=dict(size=5, color=np.linspace(0, 1, len(x_data)), colorscale='Viridis')
        )], name=str(i))
        frames.append(frame)
    fig = go.Figure(
        data=[go.Scatter3d(
            x=[CENTER], y=[CENTER], z=[1],
            mode='markers',
            marker=dict(size=5)
        )],
        layout=go.Layout(
            title="3D Diffusion-limited Aggregation",
            scene=dict(xaxis=dict(range=[0, GRID_SIZE]),
                       yaxis=dict(range=[0, GRID_SIZE]),
                       zaxis=dict(range=[0, GRID_SIZE])),
            updatemenus=[dict(type="buttons", showactive=False,
                              buttons=[
                                  dict(label="Play",
                                       method="animate",
                                       args=[None, dict(frame=dict(duration=1, redraw=True), fromcurrent=True, mode='immediate')]),
                                  dict(label="Pause",
                                       method="animate",
                                       args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')])
                              ])],
            sliders=[dict(
                active=0,
                yanchor="top",
                xanchor="left",
                currentvalue=dict(
                    font=dict(size=12),
                    prefix="Step:",
                    visible=True,
                    xanchor="right"
                ),
                transition=dict(duration=0),
                pad=dict(b=10),
                len=0.9,
                x=0.1,
                y=0,
                steps=[dict(method='animate',
                            args=[[str(i)], dict(mode='immediate', frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
                            label=str(i)) for i in range(len(frames))]
            )]
        ),
        frames=frames
    )
    fig.update_layout(dict(scene={'aspectmode':'cube'}))
    fig.write_html('RootDLA.html')

# Number of steps to simulate
num_steps = 2000
positions = simulate_dla(num_steps)
animate_dla(positions)
