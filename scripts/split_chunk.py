import open3d as o3d
import numpy as np
import sys

def chunk_pointclouds(pc1, pc2, chunk_size):
    """
    Splits two pointclouds into chunks based on their minimum and maximum values.

    Args:
    pc1: ndarray of shape (n,3), representing point cloud 1.
    pc2: ndarray of shape (m,3), representing point cloud 2.
    chunk_size: float, the size of each chunk.

    Returns:
    A list of tuples, where each tuple contains two ndarrays of shape (p,3),
    representing the chunks of pointcloud 1 and pointcloud 2 that fall within
    the same bounding box.
    """

    # # Compute the bounding box for both point clouds
    # pc1_min = np.min(pc1, axis=0)
    # pc1_max = np.max(pc1, axis=0)
    # pc2_min = np.min(pc2, axis=0)
    # pc2_max = np.max(pc2, axis=0)

    # # Compute the bounds for the chunks
    # min_bounds = np.minimum(pc1_min, pc2_min)
    # max_bounds = np.maximum(pc1_max, pc2_max)

    # Compute the bounding box for both point clouds
    pc1_min = np.min(np.asarray(pc1.points), axis=0)
    pc1_max = np.max(np.asarray(pc1.points), axis=0)
    pc2_min = np.min(np.asarray(pc2.points), axis=0)
    pc2_max = np.max(np.asarray(pc2.points), axis=0)

    # Compute the bounds for the chunks
    min_bounds = np.minimum(pc1_min, pc2_min)
    max_bounds = np.maximum(pc1_max, pc2_max)

    # Compute the number of chunks along each axis
    num_chunks_x = int(np.ceil((max_bounds[0] - min_bounds[0]) / chunk_size))
    num_chunks_y = int(np.ceil((max_bounds[1] - min_bounds[1]) / chunk_size))
    num_chunks_z = int(np.ceil((max_bounds[2] - min_bounds[2]) / chunk_size))

    # Initialize the list of chunks
    chunks = []

    # Loop over each chunk
    for i in range(num_chunks_x):
        for j in range(num_chunks_y):
            for k in range(num_chunks_z):
                # Compute the bounds of the chunk
                chunk_min = np.array([min_bounds[0] + i*chunk_size,
                                      min_bounds[1] + j*chunk_size,
                                      min_bounds[2] + k*chunk_size])
                chunk_max = np.array([min_bounds[0] + (i+1)*chunk_size,
                                      min_bounds[1] + (j+1)*chunk_size,
                                      min_bounds[2] + (k+1)*chunk_size])

                # # Extract the points within the chunk for each pointcloud
                # pc1_mask = (pc1[:,0] >= chunk_min[0]) & (pc1[:,0] < chunk_max[0]) & \
                #            (pc1[:,1] >= chunk_min[1]) & (pc1[:,1] < chunk_max[1]) & \
                #            (pc1[:,2] >= chunk_min[2]) & (pc1[:,2] < chunk_max[2])
                # pc2_mask = (pc2[:,0] >= chunk_min[0]) & (pc2[:,0] < chunk_max[0]) & \
                #            (pc2[:,1] >= chunk_min[1]) & (pc2[:,1] < chunk_max[1]) & \
                #            (pc2[:,2] >= chunk_min[2]) & (pc2[:,2] < chunk_max[2])

                # Convert point clouds to numpy arrays
                pc1_array = np.asarray(pc1.points)
                pc2_array = np.asarray(pc2.points)

                # Compute mask for chunks in pc1 and pc2
                pc1_mask = (pc1_array[:,0] >= chunk_min[0]) & (pc1_array[:,0] < chunk_max[0]) & \
                        (pc1_array[:,1] >= chunk_min[1]) & (pc1_array[:,1] < chunk_max[1]) & \
                        (pc1_array[:,2] >= chunk_min[2]) & (pc1_array[:,2] < chunk_max[2])
                pc2_mask = (pc2_array[:,0] >= chunk_min[0]) & (pc2_array[:,0] < chunk_max[0]) & \
                        (pc2_array[:,1] >= chunk_min[1]) & (pc2_array[:,1] < chunk_max[1]) & \
                        (pc2_array[:,2] >= chunk_min[2]) & (pc2_array[:,2] < chunk_max[2])

                chunk_pc1 = pc1_array[pc1_mask]
                chunk_pc2 = pc2_array[pc2_mask]

                # Append the chunk to the list of chunks
                chunks.append((chunk_pc1, chunk_pc2))

    return chunks

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

# Example usage

# size = int(sys.argv[3])

# pointcloud = o3d.io.read_point_cloud(sys.argv[1])
# pointcloud2 = o3d.io.read_point_cloud(sys.argv[2])
# # o3d.visualization.draw_geometries([pointcloud])
# chunks = split_pointcloud_uniform_fast(pointcloud, chunk_size_x=size, chunk_size_y=size, chunk_size_z=size)
# chunks2 = split_pointcloud_uniform_fast(pointcloud2, chunk_size_x=size, chunk_size_y=size, chunk_size_z=size)
# # visualize_chunks(chunks,50,50,50)
# # visualize_chunks_with_grid(pointcloud,chunk_size_x=size, chunk_size_y=size, chunk_size_z=size)

# # chunks = chunk_pointclouds(pointcloud,pointcloud2,chunk_size=10)
# # print(chunks)
# for i, chunk in enumerate(chunks):
#     o3d.io.write_point_cloud("chunk_{}.ply".format(i), chunk)

# for i, chunk in enumerate(chunks2):
#     o3d.io.write_point_cloud("chunk2_{}.ply".format(i), chunk)