import open3d as o3d
import numpy as np
import copy
import os,sys

def load_point_cloud(filename):
    pcd = o3d.io.read_point_cloud(filename)
    return pcd

# TODO Check
def addRandomPoints(point_cloud, percentage):
    """
    Effect on artifacts
    Adds a specified percentage of random points to an open3d point cloud object.
    
    Parameters:
    point_cloud (open3d.geometry.PointCloud): the point cloud object to add points to
    percentage (float): the percentage of points to add (between 0 and 1)
    
    Returns:
    point_cloud (open3d.geometry.PointCloud): the point cloud object with the added points
    """
    # Determine the number of points to add based on the percentage and the size of the existing point cloud
    num_points = int(point_cloud.points.shape[0] * percentage)
    
    # Generate random points within the bounds of the existing point cloud
    points = np.random.uniform(point_cloud.get_min_bound(), point_cloud.get_max_bound(), size=(num_points, 3))
    
    # Add the new points to the existing point cloud
    point_cloud.points = o3d.utility.Vector3dVector(np.vstack([point_cloud.points, points]))

    return point_cloud

def addGaussian(pcd,sigma_x=0.01,sigma_y=0.01,sigma_z=0.01):
    # Generate the 3D Gaussian noise and add it to the point cloud
    # Effect on accuracy
    noise = np.random.normal(0, [sigma_x, sigma_y, sigma_z], size=np.asarray(pcd.points).shape)
    noisy_pcd =  copy.deepcopy(pcd)
    noisy_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) + noise) # Update the point coordinates with noise
    return noisy_pcd


# TODO Check
def removeRandomPoints(point_cloud, percentage):
    """
    Effect on completeness
    Removes a specified percentage of random points from an open3d point cloud object.
    
    Parameters:
    point_cloud (open3d.geometry.PointCloud): the point cloud object to remove points from
    percentage (float): the percentage of points to remove (between 0 and 1)
    
    Returns:
    point_cloud (open3d.geometry.PointCloud): the point cloud object with the removed points
    """
    # Determine the number of points to remove based on the percentage and the size of the existing point cloud
    num_points = int(point_cloud.points.shape[0] * percentage)
    
    # Generate indices for random points to remove
    indices = np.random.choice(point_cloud.points.shape[0], size=num_points, replace=False)
    
    # Remove the selected points from the existing point cloud
    point_cloud.points = o3d.utility.Vector3dVector(np.delete(point_cloud.points, indices, axis=0))

    return point_cloud


# TODO Check
def uniformDownsample(point_cloud, percentage):
    """
    Effect on resolution
    Downsamples a point cloud by a specified percentage using uniform sampling.
    
    Parameters:
    point_cloud (open3d.geometry.PointCloud): the point cloud object to downsample
    percentage (float): the percentage of points to keep (between 0 and 1)
    
    Returns:
    point_cloud (open3d.geometry.PointCloud): the downsampled point cloud object
    """
    # Determine the number of points to keep based on the percentage and the size of the existing point cloud
    num_points = int(point_cloud.points.shape[0] * percentage)
    
    # Downsample the point cloud using uniform sampling
    point_cloud = point_cloud.uniform_down_sample(num_points)

    return point_cloud



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