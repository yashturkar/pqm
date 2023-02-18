import open3d as o3d
import numpy as np
import copy
import os,sys
import argparse

# python damage_pointcloud.py point_cloud.pcd gaussian --sigma_x 0.05 --sigma_y 0.05 --sigma_z 0.05 --visualize --save --output damaged_point_cloud.pcd
# python damage_pointcloud.py point_cloud.pcd random --percentage 0.9 --visualize --save --output damaged_point_cloud.pcd
# python damage_pointcloud.py point_cloud.pcd remove --percentage 0.1 --visualize --save --output damaged_point_cloud.pcd
# python damage_pointcloud.py point_cloud.pcd voxel --voxel_size 3 --visualize --save --output damaged_point_cloud.pcd
# python damage_pointcloud.py point_cloud.pcd corners --corner top_left --visualize --save --output damaged_point_cloud.pcd

def load_point_cloud(filename):
    pcd = o3d.io.read_point_cloud(filename)
    return pcd

def deleteCorners(point_cloud, corner):
    """
    Effect on completeness
    Removes a specified percentage of points from a corner of an open3d point cloud object.
    
    Parameters:
    point_cloud (open3d.geometry.PointCloud): the point cloud object to remove points from
    corner (str): the corner to remove points from (options: 'top_left', 'top_right', 'bottom_left', 'bottom_right')
    percentage (float): the percentage of points to remove (between 0 and 1)
    
    Returns:
    point_cloud (open3d.geometry.PointCloud): the point cloud object with the removed points
    """
    # Convert point_cloud to numpy array
    point_cloudnp = np.asarray(point_cloud.points)

    # Determine the bounds of the corner to remove points from
    if corner == 'top_left':
        x_min = point_cloud.get_min_bound()[0]
        x_max = point_cloud.get_center()[0]
        y_min = point_cloud.get_center()[1]
        y_max = point_cloud.get_max_bound()[1]
    elif corner == 'top_right':
        x_min = point_cloud.get_center()[0]
        x_max = point_cloud.get_max_bound()[0]
        y_min = point_cloud.get_center()[1]
        y_max = point_cloud.get_max_bound()[1]
    elif corner == 'bottom_left':
        x_min = point_cloud.get_min_bound()[0]
        x_max = point_cloud.get_center()[0]
        y_min = point_cloud.get_min_bound()[1]
        y_max = point_cloud.get_center()[1]
    elif corner == 'bottom_right':
        x_min = point_cloud.get_center()[0]
        x_max = point_cloud.get_max_bound()[0]
        y_min = point_cloud.get_min_bound()[1]
        y_max = point_cloud.get_center()[1]
    else:
        raise ValueError('Invalid corner specified.')
    
    # Determine the indices of the points within the bounds of the corner
    indices = np.where((point_cloudnp[:,0] >= x_min) & (point_cloudnp[:,0] <= x_max) & (point_cloudnp[:,1] >= y_min) & (point_cloudnp[:,1] <= y_max))[0]
    # Remove the selected points from the existing point cloud
    point_cloud.points = o3d.utility.Vector3dVector(np.delete(point_cloud.points, indices, axis=0))

    return point_cloud


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
    num_points = int(len(point_cloud.points) * percentage)
    
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
    num_points = int(len(point_cloud.points) * percentage)
    
    # Generate indices for random points to remove
    # make it 1 dimensional
    indices = np.random.choice(len(point_cloud.points), size=num_points, replace=False)
    # indices = np.random.choice(point_cloud.points, size=num_points, replace=False)
    
    # Remove the selected points from the existing point cloud
    point_cloud.points = o3d.utility.Vector3dVector(np.delete(point_cloud.points, indices, axis=0))

    return point_cloud


def voxelDownsample(point_cloud, voxel_size):
    """
    Effect on resolution
    Downsamples a point cloud by a specified voxel size using voxel downsampling.
    
    Parameters:
    point_cloud (open3d.geometry.PointCloud): the point cloud object to downsample
    voxel_size (float): the size of the voxel to use for downsampling
    
    Returns:
    point_cloud (open3d.geometry.PointCloud): the downsampled point cloud object
    """
    # Downsample the point cloud using voxel downsampling
    point_cloud = point_cloud.voxel_down_sample(voxel_size)

    return point_cloud

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
    parser = argparse.ArgumentParser(description='Damage point cloud')
    parser.add_argument('pcd', type=str, help='Path to point cloud')
    parser.add_argument('damage', type=str, help='Type of damage to apply')
    parser.add_argument('--sigma_x', type=float, help='Standard deviation of gaussian noise in x direction')
    parser.add_argument('--sigma_y', type=float, help='Standard deviation of gaussian noise in y direction')
    parser.add_argument('--sigma_z', type=float, help='Standard deviation of gaussian noise in z direction')
    parser.add_argument('--percentage', type=float, help='Percentage of points to remove')
    parser.add_argument('--corner', type=str, help='Corner to remove points from')
    parser.add_argument('--voxel_size', type=float, help='Size of voxel to use for downsampling')
    parser.add_argument('--visualize', action='store_true', help='Visualize point cloud')
    parser.add_argument('--save', action='store_true', help='Save point cloud')
    parser.add_argument('--output', type=str, help='Path to save point cloud')
    args = parser.parse_args()

    # Load point cloud
    pcd = load_point_cloud(args.pcd)

    # Damage point cloud
    if args.damage == "gaussian":
        damage_pcd = addGaussian(pcd, args.sigma_x, args.sigma_y, args.sigma_z)
    elif args.damage == "random":
        damage_pcd = addRandomPoints(pcd, args.percentage)
    elif args.damage == "remove":
        damage_pcd = removeRandomPoints(pcd, args.percentage)
    elif args.damage == "voxel":
        damage_pcd = voxelDownsample(pcd, args.voxel_size)
    elif args.damage == "corners":
        damage_pcd = deleteCorners(pcd, args.corner)

    if args.visualize:
        visualizePointcloud(damage_pcd)

    if args.save:
        savePointcloud(damage_pcd, args.output)

if __name__ == "__main__":
    main()