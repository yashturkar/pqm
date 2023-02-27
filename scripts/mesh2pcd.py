# Script to convert a mesh to a point cloud
import open3d as o3d
import numpy as np
import argparse
import os
import sys
import math


def convertMeshToPCD(mesh_path, save_path, num_points=10000):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    pcd = mesh.sample_points_poisson_disk(number_of_points=num_points)
    o3d.io.write_point_cloud(save_path, pcd)

def convertMeshToPCDBatch(mesh_path, save_path, num_points=10000, batch_size=10000):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    num_vertices = len(mesh.vertices)
    num_batches = math.ceil(num_vertices / batch_size)
    
    pcd = o3d.geometry.PointCloud()
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i+1)*batch_size, num_vertices)
        vertices_batch = mesh.vertices[start_idx:end_idx]
        pcd_batch = o3d.geometry.PointCloud()
        pcd_batch.points = o3d.utility.Vector3dVector(np.asarray(vertices_batch))
        pcd_batch = pcd_batch.sample_points_poisson_disk(number_of_points=num_points)
        pcd += pcd_batch

    o3d.io.write_point_cloud(save_path, pcd)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_path", type=str, required=True, help="Path to mesh file")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save point cloud file")
    parser.add_argument("--num_points", type=int, default=10000, help="Number of points to sample")
    args = parser.parse_args()
    convertMeshToPCD(args.mesh_path, args.save_path, args.num_points)

if __name__ == "__main__":
    main()