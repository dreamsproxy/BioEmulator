import numpy as np
import open3d as o3d
import random

class Reservoir:
    def __init__(self) -> None:
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=bool)
        self.seed = (0, CENTER, CENTER)
        self.grid[self.seed] = True

        self.translations = [
            [1, 0, 0], [-1, 0, 0], [1, 0, 1], [-1, 0, 1],
            [0, 0, 1],
            [0, 1, 0], [0, -1, 0], [0, 1, 1], [0, -1, 1],
            [0, 0, 1],
            
            [1, 1, 0], [-1, -1, 0], [1, 1, 1], [-1, -1, 1],
            [0, 0, 1],
            [-1, 1, 0], [1, -1, 0], [-1, 1, 1], [1, -1, 1],
            [0, 0, 1]
        ]

    def to_directions(self, point, empty_cells):
        directions = []
        for c in empty_cells:
            directions.append(c % point)
        return directions

    def search_points(self):
        return np.argwhere(self.grid)
    
    def empty_cells(self, point: tuple):
        x, y, z = point
        subgrid = self.grid[x-1:x+1, y-1:y+1, z-1:z+1]
        #print(subgrid)
        empty_cells = np.argwhere(subgrid == False)
        return empty_cells

    def random_walk(self):
        self.grid[(1, CENTER, CENTER)] = True
        empty_cells = self.empty_cells((1, 2, 2))
        print(empty_cells)
        directions = self.to_directions((1, 2, 2), empty_cells)
        print(directions)
        #pos = self.search_points()
        

    def visualize(self):
        points = np.argwhere(self.grid)
        points = points.astype(float)
        
        if len(points) == 0:
            print("No points to visualize.")
            return
        
        # Print the number of points to ensure points are generated
        print(f"Number of points: {len(points)}")
        
        # Create open3d point cloud
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        
        # Center the view
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(point_cloud)
        vis.get_render_option().point_size = 5.0
        
        ctr = vis.get_view_control()
        ctr.set_front([0.0, 0.0, -1.0])
        ctr.set_lookat([GRID_SIZE // 2, GRID_SIZE // 2, GRID_SIZE // 2])
        ctr.set_up([0.0, -1.0, 0.0])
        ctr.set_zoom(0.8)
        
        vis.run()
        vis.destroy_window()

GRID_SIZE = 32
CENTER = GRID_SIZE // 2

grid = Reservoir()
grid.random_walk()
#grid.visualize()