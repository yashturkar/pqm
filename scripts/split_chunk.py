import open3d as o3d
import numpy as np

# def split_pointcloud(pointcloud, chunk_size_x, chunk_size_y):
#     points = np.asarray(pointcloud.points)
#     x_min, y_min, z_min = np.min(points, axis=0)
#     x_max, y_max, z_max = np.max(points, axis=0)
#     num_chunks_x = int(np.ceil((x_max - x_min) / chunk_size_x))
#     num_chunks_y = int(np.ceil((y_max - y_min) / chunk_size_y))
#     chunk_size_z = (z_max - z_min) / max(num_chunks_x, num_chunks_y)
    
#     chunks = []
#     for i in range(num_chunks_x):
#         for j in range(num_chunks_y):
#             x_start = x_min + i * chunk_size_x
#             x_end = x_start + chunk_size_x
#             y_start = y_min + j * chunk_size_y
#             y_end = y_start + chunk_size_y
#             chunk = pointcloud.crop(
#                 o3d.geometry.AxisAlignedBoundingBox(
#                     min_bound=[x_start, y_start, z_min],
#                     max_bound=[x_end, y_end, z_min + chunk_size_z]
#                 )
#             )
#             chunks.append(chunk)
#     return chunks

def split_pointcloud_uniform(pointcloud, chunk_size_x, chunk_size_y, chunk_size_z):
    points = np.asarray(pointcloud.points)
    x_min, y_min, z_min = np.min(points, axis=0)
    x_max, y_max, z_max = np.max(points, axis=0)
    num_chunks_x = int(np.ceil((x_max - x_min) / chunk_size_x))
    num_chunks_y = int(np.ceil((y_max - y_min) / chunk_size_y))
    num_chunks_z = int(np.ceil((z_max - z_min) / chunk_size_z))
    chunk_size_x = (x_max - x_min) / num_chunks_x
    chunk_size_y = (y_max - y_min) / num_chunks_y
    chunk_size_z = (z_max - z_min) / num_chunks_z
    
    chunks = []
    for i in range(num_chunks_x):
        for j in range(num_chunks_y):
            for k in range(num_chunks_z):
                x_start = x_min + i * chunk_size_x
                x_end = x_start + chunk_size_x
                y_start = y_min + j * chunk_size_y
                y_end = y_start + chunk_size_y
                z_start = z_min + k * chunk_size_z
                z_end = z_start + chunk_size_z
                chunk_bounds = o3d.geometry.AxisAlignedBoundingBox(
                    min_bound=[x_start, y_start, z_start],
                    max_bound=[x_end, y_end, z_end]
                )
                chunk = pointcloud.crop(chunk_bounds)
                if chunk.has_points:
                    chunks.append(chunk)
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

# Example usage
pointcloud = o3d.io.read_point_cloud("/home/yashturkar/Syncspace/final_results_reg/Mai_City/Puma/Puma_3_100k.ply")
# o3d.visualization.draw_geometries([pointcloud])
chunks = split_pointcloud_uniform_fast(pointcloud, chunk_size_x=50, chunk_size_y=50, chunk_size_z=50)
for i, chunk in enumerate(chunks):
    o3d.io.write_point_cloud("chunk_{}.ply".format(i), chunk)