import numpy as np
import open3d as o3d

class Chunk:
    def __init__(self, pointcloud_GT, pointcloud_Cnd, x_min, y_min, z_min, size):
        self.pointcloud_GT = pointcloud_GT
        self.pointcloud_Cnd = pointcloud_Cnd
        self.completeness = None
        self.accuracy = None
        self.artifacts = None
        self.resolution = None
        self.quality = None
        self.x_min = x_min
        self.y_min = y_min
        self.z_min = z_min
        self.size = size
        self.color = None
        self.empty = False
        
    def check_empty(self):
        # check if PCD is empty
        pass
    def update_completeness(self, value):
        self.completeness = value
        
    def update_accuracy(self, value):
        self.accuracy = value
        
    def update_artifacts(self, value):
        self.artifacts = value
        
    def update_resolution(self, value):
        self.resolution = value
        
    def update_quality(self, value):
        self.quality = value
        
    def viz(self):
        pass


def split_pointcloud_uniform_fast(pointcloud, chunk_size_x, chunk_size_y, chunk_size_z):
    points = np.asarray(pointcloud.points)
    x_min, y_min, z_min = np.min(points, axis=0)
    x_max, y_max, z_max = np.max(points, axis=0)
    num_chunks_x = int(np.ceil((x_max - x_min) / chunk_size_x))
    num_chunks_y = int(np.ceil((y_max - y_min) / chunk_size_y))
    num_chunks_z = int(np.ceil((z_max - z_min) / chunk_size_z))
    chunk_size_x = (x_max - x_min) / num_chunks_x
    chunk_size_y = (y_max - y_min) / num_chunks_y
    chunk_size_z = (z_max - z_min) / num_chunks_z
    
    x_indices = np.floor((points[:, 0] - x_min) / chunk_size_x).astype(int)
    y_indices = np.floor((points[:, 1] - y_min) / chunk_size_y).astype(int)
    z_indices = np.floor((points[:, 2] - z_min) / chunk_size_z).astype(int)
    indices = num_chunks_y * num_chunks_z * x_indices + num_chunks_z * y_indices + z_indices
    
    chunks = []
    for i in range(num_chunks_x * num_chunks_y * num_chunks_z):
        chunk_indices = np.where(indices == i)
        if chunk_indices[0].size > 0:
            chunk = o3d.geometry.PointCloud()
            chunk.points = o3d.utility.Vector3dVector(points[chunk_indices])
            chunks.append(chunk)
    return chunks

def visualize_chunks(chunks, chunk_size_x, chunk_size_y, chunk_size_z):
    """
    Visualize chunks in one point cloud with a grid-like view.

    Args:
        chunks: List of point clouds representing the chunks.
        chunk_size_x: Size of chunks along the x-axis.
        chunk_size_y: Size of chunks along the y-axis.
        chunk_size_z: Size of chunks along the z-axis.
    """
    merged_cloud = o3d.geometry.PointCloud()
    colors = [[0.5, 0.5, 0.5] for _ in range(len(chunks))]
    merged_cloud.colors = o3d.utility.Vector3dVector(colors)

    num_chunks_x = int(np.ceil((np.max([np.max(np.asarray(chunk.points), axis=0) for chunk in chunks], axis=0)[0] - np.min([np.min(np.asarray(chunk.points), axis=0) for chunk in chunks], axis=0)[0]) / chunk_size_x))
    num_chunks_y = int(np.ceil((np.max([np.max(np.asarray(chunk.points), axis=0) for chunk in chunks], axis=0)[1] - np.min([np.min(np.asarray(chunk.points), axis=0) for chunk in chunks], axis=0)[1]) / chunk_size_y))
    num_chunks_z = int(np.ceil((np.max([np.max(np.asarray(chunk.points), axis=0) for chunk in chunks], axis=0)[2] - np.min([np.min(np.asarray(chunk.points), axis=0) for chunk in chunks], axis=0)[2]) / chunk_size_z))

    for i in range(len(chunks)):
        chunk = chunks[i]
        # x_offset = int(i / (num_chunks_y * num_chunks_z)) * chunk_size_x
        # y_offset = int((i / num_chunks_z) % num_chunks_y) * chunk_size_y
        # z_offset = int(i % num_chunks_z) * chunk_size_z
        x_offset = 0
        y_offset = 0
        z_offset = 0
        offset = np.array([x_offset, y_offset, z_offset])
        chunk_points = np.asarray(chunk.points)
        chunk_points += offset

        merged_cloud += chunk

    o3d.visualization.draw_geometries([merged_cloud])


def visualize_chunks_with_grid(pointcloud, chunk_size_x=None, chunk_size_y=None, chunk_size_z=None, chunks=None):
    # Determine chunk sizes
    if chunk_size_x is not None and chunk_size_y is not None and chunk_size_z is not None:
        x_size, y_size, z_size = chunk_size_x, chunk_size_y, chunk_size_z
    elif chunks is not None:
        # Infer chunk size from first chunk
        x_size, y_size, z_size = np.max(chunks[0].get_max_bound() - chunks[0].get_min_bound(), axis=0) / 2
    else:
        raise ValueError("Either chunk size or chunks must be provided.")

    # Split point cloud into chunks
    if chunks is None:
        chunks = split_pointcloud_uniform_fast(pointcloud, x_size, y_size, z_size)
    
    x_min, y_min, z_min = np.min(np.asarray(pointcloud.points), axis=0)
    x_max, y_max, z_max = np.max(np.asarray(pointcloud.points), axis=0)
    
    # Visualize chunks with grid
    pcd_combined = o3d.geometry.PointCloud()
    for i, chunk in enumerate(chunks):
        color = [0, 0, i / len(chunks)]
        chunk.paint_uniform_color(color)
        pcd_combined += chunk

    pcd_min = pcd_combined.get_min_bound()
    pcd_max = pcd_combined.get_max_bound()

    x_range = np.arange(np.floor(pcd_min[0] / x_size) * x_size, np.ceil(pcd_max[0] / x_size) * x_size, x_size)
    y_range = np.arange(np.floor(pcd_min[1] / y_size) * y_size, np.ceil(pcd_max[1] / y_size) * y_size, y_size)
    z_range = np.arange(np.floor(pcd_min[2] / z_size) * z_size, np.ceil(pcd_max[2] / z_size) * z_size, z_size)

    grid_lines = []
    for x in x_range:
        for y in y_range:
            line = o3d.geometry.LineSet()
            points = [[x, y, z_min], [x, y, z_max]]
            line.points = o3d.utility.Vector3dVector(points)
            line.lines = o3d.utility.Vector2iVector([[0, 1]])
            line.paint_uniform_color([1, 0, 0])
            grid_lines.append(line)

    for y in y_range:
        for z in z_range:
            line = o3d.geometry.LineSet()
            points = [[x_min, y, z], [x_max, y, z]]
            line.points = o3d.utility.Vector3dVector(points)
            line.lines = o3d.utility.Vector2iVector([[0, 1]])
            line.paint_uniform_color([0, 1, 0])
            grid_lines.append(line)

    for x in x_range:
        for z in z_range:
            line = o3d.geometry.LineSet()
            points = [[x, y_min, z], [x, y_max, z]]
            line.points = o3d.utility.Vector3dVector(points)
            line.lines = o3d.utility.Vector2iVector([[0, 1]])
            line.paint_uniform_color([0, 0, 1])
            grid_lines.append(line)

    o3d.visualization.draw_geometries([pcd_combined, *grid_lines])
