import open3d as o3d
import numpy as np
import sys,os
import copy

# Load a point cloud from file
pcd = o3d.io.read_point_cloud(sys.argv[1])


def addGaussian(pcd,sigma_x=0.01,sigma_y=0.01,sigma_z=0.01):
    # Generate the 3D Gaussian noise and add it to the point cloud
    noise = np.random.normal(0, [sigma_x, sigma_y, sigma_z], size=np.asarray(pcd.points).shape)
    noisy_pcd =  copy.deepcopy(pcd)
    noisy_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) + noise) # Update the point coordinates with noise
    return noisy_pcd


if __name__ == "__main__":

    noisy_pcd = addGaussian(pcd)

    # Define colors for the two point clouds
    color_orig = [1, 0, 0] # red for original point cloud
    color_noisy = [0, 1, 0] # green for noisy point cloud

    # Assign colors to the two point clouds
    pcd.paint_uniform_color(color_orig)
    noisy_pcd.paint_uniform_color(color_noisy)

    # Create a new point cloud by appending the original and noisy point clouds
    pcd_combined = pcd + noisy_pcd

    # Visualize the combined point cloud with colored points
    o3d.visualization.draw_geometries([pcd_combined])