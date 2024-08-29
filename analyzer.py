import numpy as np
import cv2
import collections
from itertools import islice
import plotly.graph_objects as go
import matplotlib.cm as cm

def sliding_window(iterable, n):
    window = collections.deque(islice(iterable, n - 1), maxlen=n)
    for x in iterable[n-1:]:
        window.append(x)
        yield tuple(window)

# Load and process the image
img = cv2.imread('./Enchinodermata.png', cv2.IMREAD_GRAYSCALE)
img = img[:450, 25:475]  # Crop the image
img = cv2.resize(img, (256, 256))  # Resize the image
img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)  # Normalize

def batched(iterable, n):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        yield batch

alpha_windows = batched([i for i in range(255)], 8)

# Initialize plotly figure
fig = go.Figure()

# Generate the layers
for idx, alpha_group in enumerate(alpha_windows):
    start = alpha_group[0]
    end = alpha_group[-1]
    mask = (img >= start) & (img <= end)
    
    # Get the indices of the pixels in the mask
    y, x = np.where(mask)
    
    # Use Viridis colormap for color mapping
    norm_intensity = (img[mask] - start) / (end - start)  # Normalize to [0, 1] within the range
    colors = cm.viridis(norm_intensity)
    
    # Create a 3D scatter plot for this layer
    scatter = go.Scatter3d(
        x=x,
        y=y,
        z=np.full_like(x, idx),  # The z-coordinate is the layer index
        mode='markers',
        marker=dict(
            size=2,
            color=colors[:, :3],  # Use RGB values from the colormap
            opacity=0.6
        )
    )
    
    fig.add_trace(scatter)

# Update layout to make the plot visually appealing
fig.update_layout(
    scene=dict(
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        zaxis=dict(title='Layer Index'),
    ),
    title="3D Scatter Plot of Image Layers",
)

fig.write_html('Layers.html')
