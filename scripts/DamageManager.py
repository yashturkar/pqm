import open3d as o3d
import numpy as np
import copy
import os,sys
import argparse

from system_constants import *

from util import get_cropping_bound, get_cropped_point_cloud, generate_grid_lines

# python damage_pointcloud.py point_cloud.pcd gaussian --sigma_x 0.05 --sigma_y 0.05 --sigma_z 0.05 --visualize --save --output damaged_point_cloud.pcd
# python damage_pointcloud.py point_cloud.pcd random --percentage 0.9 --visualize --save --output damaged_point_cloud.pcd
# python damage_pointcloud.py point_cloud.pcd remove --percentage 0.1 --visualize --save --output damaged_point_cloud.pcd
# python damage_pointcloud.py point_cloud.pcd voxel --voxel_size 3 --visualize --save --output damaged_point_cloud.pcd
# python damage_pointcloud.py point_cloud.pcd corners --corner top_left --visualize --save --output damaged_point_cloud.pcd



def load_point_cloud(filename):
    pcd = o3d.io.read_point_cloud(filename)
    return pcd

class DamageManager:
    def __init__(self, point_cloud, epislon, cell_size=0.1):
        self.point_cloud = point_cloud
        self.point_cloud.paint_uniform_color(GT_COLOR)


        self.cell_size = cell_size
        self.epislon = epislon
        #compute the min bound of the pointcloud
        bb1 = self.point_cloud.get_axis_aligned_bounding_box()
        self.min_bound = bb1.min_bound
        self.max_bound = bb1.max_bound

        #print(self.min_bound, self.max_bound)

        self.cell_dim = (np.ceil((self.max_bound - self.min_bound) / self.cell_size)).astype(int)
        self.max_bound = self.min_bound + self.cell_dim * self.cell_size 
        self.damage_fn_map = {
    "gaussian": self.addGaussian,
    "add": self.addPoints_Sphere,
    "remove": self.removeRandomPoints,
    "voxel": self.voxelDownsample,
    "corners": self.deleteCorners
        }
        

    def iterate_cells(self):
        #iterate through all the Cells
        for i in range(int(self.cell_dim[0])):
            for j in range(int(self.cell_dim[1])):
                for k in range(int(self.cell_dim[2])):
                    min_cell_index = np.array([i, j, k])
                    max_cell_index = np.array([i+1, j+1, k+1])
                    #print(min_cell_index, max_cell_index)
                    yield min_cell_index, max_cell_index


    def damage_point_cloud(self, damage_type, damage_parameters):
        print('Damage type: ', damage_type)
        print('Damage parameters: ', damage_parameters)

        
        damaged_point_cloud = copy.deepcopy(self.point_cloud)

        if damage_type not in self.damage_fn_map:
            raise ValueError('Damage type not recognized: ', damage_type)
        
        damage_fn = self.damage_fn_map[damage_type]
        
        damaged_point_cloud = self.damage_per_cell(damage_fn, damage_parameters)
        return damaged_point_cloud

    def damage_per_cell(self, damage_fn, damage_parameters):
        #iterate through all the Cells
        damaged_pcd_final = o3d.geometry.PointCloud()
        print("cell_dim: ", self.cell_dim)
        for min_cell_index, max_cell_index in self.iterate_cells():
            damaged_cell_points = damage_fn(min_cell_index, max_cell_index, damage_parameters)
            damaged_pcd_final.points = o3d.utility.Vector3dVector(np.vstack([damaged_pcd_final.points, damaged_cell_points.points]))
        
        return damaged_pcd_final


    def deleteCorners(self, min_cell_index, max_cell_index, corner):
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
        damage_pcd, _ = get_cropped_point_cloud(self.point_cloud, self.min_bound, self.cell_size, min_cell_index, max_cell_index)

        # Convert point_cloud to numpy array
        point_cloudnp = np.asarray(damage_pcd.points)

        # Determine the bounds of the corner to remove points from
        if corner == 'top_left':
            x_min = damage_pcd.get_min_bound()[0]
            x_max = damage_pcd.get_center()[0]
            y_min = damage_pcd.get_center()[1]
            y_max = damage_pcd.get_max_bound()[1]
        elif corner == 'top_right':
            x_min = damage_pcd.get_center()[0]
            x_max = damage_pcd.get_max_bound()[0]
            y_min = damage_pcd.get_center()[1]
            y_max = damage_pcd.get_max_bound()[1]
        elif corner == 'bottom_left':
            x_min = damage_pcd.get_min_bound()[0]
            x_max = damage_pcd.get_center()[0]
            y_min = damage_pcd.get_min_bound()[1]
            y_max = damage_pcd.get_center()[1]
        elif corner == 'bottom_right':
            x_min = damage_pcd.get_center()[0]
            x_max = damage_pcd.get_max_bound()[0]
            y_min = damage_pcd.get_min_bound()[1]
            y_max = damage_pcd.get_center()[1]
        else:
            raise ValueError('Invalid corner specified.')
        
        # Determine the indices of the points within the bounds of the corner
        indices = np.where((point_cloudnp[:,0] >= x_min) & (point_cloudnp[:,0] <= x_max) & (point_cloudnp[:,1] >= y_min) & (point_cloudnp[:,1] <= y_max))[0]
        # Remove the selected points from the existing point cloud
        damage_pcd.points = o3d.utility.Vector3dVector(np.delete(damage_pcd.points, indices, axis=0))

        return damage_pcd


    def get_cell_bound(self, cell_index):

        cell_index_vals = cell_index
        cell_index_next = cell_index_vals + np.array([1, 1, 1])

        min_bound, max_bound = get_cropping_bound(self.min_bound, self.cell_size, cell_index_vals, cell_index_next)
        return min_bound, max_bound
    

    def addPoints_Sphere(self, min_cell_index, max_cell_index, percentage):
        """
        Effect on artifacts
        Adds a specified percentage of random points to an open3d point cloud object.
        
        Parameters:
        point_cloud (open3d.geometry.PointCloud): the point cloud object to add points to
        percentage (float): the percentage of points to add (between 0 and 1)
        
        Returns:
        point_cloud (open3d.geometry.PointCloud): the point cloud object with the added points
        """
        
        damage_pcd, bbox = get_cropped_point_cloud(self.point_cloud, self.min_bound, self.cell_size, min_cell_index, max_cell_index)
        #print("cell index", min_cell_index, max_cell_index, self.cell_size)

        # Determine the number of points to add based on the percentage and the size of the existing point cloud
        num_points = int(len(damage_pcd.points) * percentage)

        if num_points == 0:
            return damage_pcd
        # Generate a sphere mesh to use as a template for the new points
        sphere_radius = self.cell_size/10
        sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)

        translate_center = bbox.get_min_bound()+np.array([self.cell_size, self.cell_size, self.cell_size])/2
        #min_bound, max_bound = self.get_cell_bound()
        #print("sphere_center", translate_center)
        
        sphere_mesh.translate(translate_center)

        sphere_pcd = sphere_mesh.sample_points_uniformly(number_of_points=num_points)

        #damage_pcd += sphere_pcd
        # # Generate random points within the bounds of the existing point cloud
        # points = np.random.uniform(damage_pcd.get_min_bound(), damage_pcd.get_max_bound(), size=(num_points, 3))
        
        # Add the new points to the existing point cloud
        damage_pcd.points = o3d.utility.Vector3dVector(np.vstack([damage_pcd.points, sphere_pcd.points]))

        return damage_pcd

    def addGaussian(self, min_cell_index, max_cell_index, percentage):#_x=0.01,sigma_y=0.01,sigma_z=0.01):
        # Generate the 3D Gaussian noise and add it to the point cloud
        # Effect on accuracy
        damage_pcd, _ = get_cropped_point_cloud(self.point_cloud, self.min_bound, self.cell_size, min_cell_index, max_cell_index)
        sigma = self.epislon/2.0

        num_points = int(len(damage_pcd.points) * percentage)

        if num_points == 0:
            return damage_pcd
        
        # select non-affected points indices
        indices = np.random.choice(len(damage_pcd.points), size=(len(damage_pcd.points)-num_points), replace=False)
        
        noise = np.random.normal(0, [sigma/2.0, sigma/2.0, sigma/2.0], size=np.asarray(damage_pcd.points).shape)

        noise[indices, :] = [0,0,0]

        damage_pcd.points = o3d.utility.Vector3dVector(np.asarray(damage_pcd.points) + noise) # Update the point coordinates with noise
        return damage_pcd

    def addGaussian_old(self, min_cell_index, max_cell_index, sigma):#_x=0.01,sigma_y=0.01,sigma_z=0.01):
        # Generate the 3D Gaussian noise and add it to the point cloud
        # Effect on accuracy
        damage_pcd, _ = get_cropped_point_cloud(self.point_cloud, self.min_bound, self.cell_size, min_cell_index, max_cell_index)

        noise = np.random.normal(0, [sigma, sigma, sigma], size=np.asarray(damage_pcd.points).shape)
        damage_pcd.points = o3d.utility.Vector3dVector(np.asarray(damage_pcd.points) + noise) # Update the point coordinates with noise
        return damage_pcd


    def find_Knearest_points(self, pcd, point, k):
        # Find the k nearest points to each point in the point cloud
        # Effect on accuracy
        #print("Find {}-nearest neighbors".format(k))
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        [_, idx, _] = pcd_tree.search_knn_vector_3d(point, k)
        return idx
    

    def removeRandomPoints(self, min_cell_index, max_cell_index, percentage):
        """
        Effect on completeness
        Removes a specified percentage of random points from an open3d point cloud object.
        
        Parameters:
        point_cloud (open3d.geometry.PointCloud): the point cloud object to remove points from
        percentage (float): the percentage of points to remove (between 0 and 1)
        
        Returns:
        point_cloud (open3d.geometry.PointCloud): the point cloud object with the removed points
        """
        
        damage_pcd, bbox= get_cropped_point_cloud(self.point_cloud, self.min_bound, self.cell_size, min_cell_index, max_cell_index)
        
        if len(damage_pcd.points) == 0:
            return damage_pcd

        # Determine the number of points to remove based on the percentage and the size of the existing point cloud
        #print("center", bbox.get_center())
        num_points = int(len(damage_pcd.points) * percentage)

        if num_points == 0:
            return damage_pcd
        
        index = np.random.choice(len(damage_pcd.points), size=1, replace=False)[0]

        index = self.find_Knearest_points(damage_pcd, bbox.get_center(), 1)[0]

        selected_point = damage_pcd.points[index]

        indices = self.find_Knearest_points(damage_pcd, selected_point, num_points)
        
        # Remove the selected points from the existing point cloud
        damage_pcd.points = o3d.utility.Vector3dVector(np.delete(damage_pcd.points, indices, axis=0))

        return damage_pcd


    def voxelDownsample(self, min_cell_index, max_cell_index, voxel_size):
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
        damage_pcd, _ = get_cropped_point_cloud(self.point_cloud, self.min_bound, self.cell_size, min_cell_index, max_cell_index)
        #damage_pcd = damage_pcd.voxel_down_sample(voxel_size)
        #print("voxel size", voxel_size)
        damage_pcd = damage_pcd.uniform_down_sample(voxel_size)

        return damage_pcd

    def visualizePointcloud(self, pcd):
        # Visualize the point cloud
        o3d.visualization.draw_geometries([pcd])

    def visualizePointcloudSeparate(self, damaged_pcd):
        # Visualize the point cloud
        # Define colors for the two point clouds
        color_orig =  GT_COLOR # green for original point cloud
        color_noisy = CND_COLOR # red for noisy point cloud

        # Assign colors to the two point clouds
        self.point_cloud.paint_uniform_color(color_orig)
        damaged_pcd.paint_uniform_color(color_noisy)

        # Visualize the combined point cloud with colored points
        o3d.visualization.draw_geometries([self.point_cloud, damaged_pcd])

    def savePointcloud(self, pcd, filename):
        # Save the point cloud
        o3d.io.write_point_cloud(filename, pcd)