import numpy as np
import open3d as o3d

import copy
#import open3d.pipelines.registration as treg

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])
class MapMetricManager:
    def __init__(self, pointcloud_GT, pointcloud_Cnd, chunk_size):
        self.pointcloud_GT = pointcloud_GT
        self.pointcloud_Cnd = pointcloud_Cnd
        self.chunk_size = chunk_size
        #compute the min bound of the pointcloud
        bb1 = self.pointcloud_Cnd.get_axis_aligned_bounding_box()
        bb2 = self.pointcloud_GT.get_axis_aligned_bounding_box()
        # print(bb1.min_bound, bb1.max_bound)
        # print(bb2.min_bound, bb2.max_bound)

        self.min_bound = np.minimum(bb1.min_bound, bb2.min_bound)
        self.max_bound = np.maximum(bb1.max_bound, bb2.max_bound)

        print(self.min_bound, self.max_bound)

        self.cell_x_size = np.ceil((self.max_bound[0] - self.min_bound[0]) / self.chunk_size)
        self.cell_y_size = np.ceil((self.max_bound[1] - self.min_bound[1]) / self.chunk_size)
        self.cell_z_size = np.ceil((self.max_bound[2] - self.min_bound[2]) / self.chunk_size)

        print(self.cell_x_size, self.cell_y_size, self.cell_z_size)
        # n=2
        # cell_bound = [self.min_bound[0]+ chunk_size*7, self.min_bound[1]+chunk_size*11, self.min_bound[2]+chunk_size*n]
        # print(cell_bound)
        # bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=self.min_bound, max_bound=cell_bound)
        # self.cropped_GT = self.pointcloud_GT.crop(bbox)



    #visualize pointcloud
    def visualize(self):
        #visualize the pointcloud
        #o3d.visualization.draw_geometries([self.pointcloud_GT, self.pointcloud_Cnd])
        o3d.visualization.draw_geometries([self.pointcloud_GT, self.pointcloud_Cnd])

    #visualize pointcloud GT
    def visualize_GT(self):
        #visualize the pointcloud
        o3d.visualization.draw_geometries([self.pointcloud_GT])

    #visualize pointcloud Cnd
    def visualize_Cnd(self):
        #visualize the pointcloud
        o3d.visualization.draw_geometries([self.pointcloud_Cnd])

    #visualize pointcloud with grid
    def visualize_sub_point_cloud(self, chunk_size, min_cell_index, max_cell_index, save=False, filename="test.pcd"):
        #visualize the pointcloud with grid
        #o3d.visualization.draw_geometries([self.pointcloud_GT, self.pointcloud_Cnd])
        min_bound = [self.min_bound[0]+ chunk_size*min_cell_index[0], self.min_bound[1]+chunk_size*min_cell_index[1], self.min_bound[2]+chunk_size*min_cell_index[2]]
        max_bound = [self.min_bound[0]+ chunk_size*max_cell_index[0], self.min_bound[1]+chunk_size*max_cell_index[1], self.min_bound[2]+chunk_size*max_cell_index[2]]

        print(min_bound, max_bound)

        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
        cropped_gt = self.pointcloud_GT.crop(bbox)
        cropped_gt.paint_uniform_color([1, 0, 0])
        cropped_candidate = self.pointcloud_Cnd.crop(bbox)
        cropped_candidate.paint_uniform_color([0, 1, 0])
        
        if save:
            o3d.io.write_point_cloud(filename+"_gt.pcd", cropped_gt)
            o3d.io.write_point_cloud(filename+"_cnd.pcd", cropped_candidate)
        o3d.visualization.draw_geometries([cropped_gt, cropped_candidate])



    def visualize_registered_point_cloud(self):
        # register two pointcloud
        trans_init = np.eye(4)

        # trans_init = np.array([[1.000	,-0.008	,0.000	,1.095],
        #                         [0.008	,1.000	,-0.008	,0.083],
        #                         [-0.000	,0.008	,1.000	,-1.183],
        #                         [0.000	,0.000	,0.000	,1.000]])
        
        # trans_init = np.array([[1.000,	0.013,	-0.005,	-0.873],
        #                         [-0.013,	1.000,	-0.001,	0.070],
        #                         [0.005,	0.001	,1.000,	1.251],
        #                         [0.000,	0.000,	0.000	,1.000]])


        draw_registration_result( self.pointcloud_Cnd,self.pointcloud_GT, trans_init)

        threshold = 0.02

        mu, sigma = 0, 0.5  # mean and standard deviation
        print("Robust point-to-plane ICP, threshold={}:".format(threshold))
        loss = o3d.pipelines.registration.TukeyLoss(k=sigma)
        print("Using robust loss:", loss)

        self.pointcloud_Cnd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        self.pointcloud_GT.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        p2l = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
        reg_p2l = o3d.pipelines.registration.registration_icp(self.pointcloud_Cnd, self.pointcloud_GT,
                                                            threshold, trans_init,
                                                            p2l)
        print(reg_p2l)
        print("Transformation is:")
        print(reg_p2l.transformation)
        draw_registration_result(self.pointcloud_Cnd, self.pointcloud_GT, reg_p2l.transformation)




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
        return self.empty
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

import sys

import argparse


def main():

    # Example usage
    # argument argparse following variables
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=str, help="GT point cloud")
    parser.add_argument("--cnd", type=str, help="Cnd point cloud")
    parser.add_argument("--sub_sample", help="sub sample size", action="store_true")
    parser.add_argument("--save", help="save file name", action="store_true")
    parser.add_argument("--filename", type=str, help="file name" , default="test_subsample")
    parser.add_argument("--size", type=int, help="sub sample size", default=10)

    parser.add_argument("--min_cell", type=str, help="Min cell index i.e. 0,0,0 or higher", default="2,2,0")
    parser.add_argument("--max_cell", type=str, help="Max cell index i.e. 1,1,1 or higher",default="4,4,1")

    parser.add_argument("--register", help="register point cloud", action="store_true")


    args = parser.parse_args()
    

    min_cell = [int(item) for item in args.min_cell.split(',')]
    max_cell = [int(item) for item in args.max_cell.split(',')]

    pointcloud = o3d.io.read_point_cloud(args.gt)
    pointcloud2 = o3d.io.read_point_cloud(args.cnd)

    mapManager = MapMetricManager(pointcloud,pointcloud2, args.size)

    if args.register:
        mapManager.visualize_registered_point_cloud()
    elif args.sub_sample:
        mapManager.visualize_sub_point_cloud(args.size, min_cell, max_cell, save=args.save, filename=args.filename)
    else:
        draw_registration_result(pointcloud,pointcloud2, np.eye(4))
        #mapManager.visualize_GT()
        #mapManager.visualize_Cnd()


#main entry point
if __name__ == "__main__":

    main()