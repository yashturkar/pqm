import open3d as o3d
import numpy as np
import copy
import os,sys

def load_point_cloud(filename):
    pcd = o3d.io.read_point_cloud(filename)
    return pcd

def addGaussian(pcd,sigma_x=0.01,sigma_y=0.01,sigma_z=0.01):
    # Generate the 3D Gaussian noise and add it to the point cloud
    noise = np.random.normal(0, [sigma_x, sigma_y, sigma_z], size=np.asarray(pcd.points).shape)
    noisy_pcd =  copy.deepcopy(pcd)
    noisy_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) + noise) # Update the point coordinates with noise
    return noisy_pcd

def removeRandomPoints(pcd, percentage):
    # Remove random points
    pcd_down = pcd.voxel_down_sample(voxel_size=0.05)
    points = np.asarray(pcd_down.points)
    points = points[::percentage]
    pcd_down.points = o3d.utility.Vector3dVector(points)
    return pcd_down

def damage_point_cloud(pcd, sigma_x=0.01, sigma_y=0.01, sigma_z=0.01, percentage=10):
    # Add noise and remove random points
    noisy_pcd = addGaussian(pcd, sigma_x, sigma_y, sigma_z)
    damaged_pcd = removeRandomPoints(noisy_pcd, percentage)
    return damaged_pcd

def visualizePointcloud(pcd):
    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])

def visualizePointcloudSeparate(pcd, damaged_pcd):
    # Visualize the point cloud
    # Define colors for the two point clouds
    color_orig = [1, 0, 0] # red for original point cloud
    color_noisy = [0, 1, 0] # green for noisy point cloud

    # Assign colors to the two point clouds
    pcd.paint_uniform_color(color_orig)
    damaged_pcd.paint_uniform_color(color_noisy)

    # Visualize the combined point cloud with colored points
    o3d.visualization.draw_geometries([pcd, damaged_pcd])

def savePointcloud(pcd, filename):
    # Save the point cloud
    o3d.io.write_point_cloud(filename, pcd)



def main():
    # Load point cloud
    pcd = load_point_cloud(sys.argv[1])

    # Damage point cloud
    damaged_pcd = damage_point_cloud(pcd, sigma_x=0.05, sigma_y=0.05, sigma_z=0.05, percentage=50)

    savePointcloud(damaged_pcd, sys.argv[2])

    # Visualize the point cloud
    # visualizePointcloud(damaged_pcd)
    visualizePointcloudSeparate(pcd, damaged_pcd)

if __name__ == "__main__":
    main()