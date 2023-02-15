import numpy as np
import open3d as o3d

import copy

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
    

def apply_noise(pcd, mu, sigma):
    noisy_pcd = copy.deepcopy(pcd)
    points = np.asarray(noisy_pcd.points)
    points += np.random.normal(mu, sigma, size=points.shape)
    noisy_pcd.points = o3d.utility.Vector3dVector(points)
    return noisy_pcd

def get_cropping_bound(min_bound, chunk_size, min_cell_index, max_cell_index):
    min_bound = [min_bound[0]+ chunk_size*min_cell_index[0], min_bound[1]+chunk_size*min_cell_index[1], min_bound[2]+chunk_size*min_cell_index[2]]
    max_bound = [min_bound[0]+ chunk_size*max_cell_index[0], min_bound[1]+chunk_size*max_cell_index[1], min_bound[2]+chunk_size*max_cell_index[2]]
    return min_bound, max_bound

def get_cropped_point_cloud(pcd, min_bound_source, chunk_size, min_cell_index, max_cell_index):
    min_bound, max_bound = get_cropping_bound(min_bound_source, chunk_size, min_cell_index, max_cell_index)
    #print(min_bound, max_bound)
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
    pcd_cropped = pcd.crop(bbox)
    return pcd_cropped, bbox


def generate_noisy_point_cloud(pcd, sigma, filename="test"):

    noisy_gt = apply_noise(pcd, 0, sigma)
    o3d.io.write_point_cloud(filename+"_gt_noisy.pcd", noisy_gt)
    o3d.visualization.draw_geometries([noisy_gt, pcd])




def generate_grid_lines(min_bound, max_bound, cell_count):
    print(min_bound, max_bound, cell_count)
    
    x_range = np.linspace(min_bound[0], max_bound[0], cell_count[0]+1)
    y_range = np.linspace(min_bound[1], max_bound[1], cell_count[1]+1)
    z_range = np.linspace(min_bound[2], max_bound[2], cell_count[2]+1)

    x_min, y_min, z_min = min_bound
    x_max, y_max, z_max = max_bound
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
    return grid_lines
















def visualize_registered_point_cloud(pcd1, pcd2):
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


    draw_registration_result( pcd1,pcd2, trans_init)

    threshold = 0.02

    mu, sigma = 0, 0.5  # mean and standard deviation
    print("Robust point-to-plane ICP, threshold={}:".format(threshold))
    loss = o3d.pipelines.registration.TukeyLoss(k=sigma)
    print("Using robust loss:", loss)

    pcd1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    p2l = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
    reg_p2l = o3d.pipelines.registration.registration_icp(pcd1, pcd2,
                                                        threshold, trans_init,
                                                        p2l)
    print(reg_p2l)
    print("Transformation is:")
    print(reg_p2l.transformation)
    draw_registration_result(pcd1, pcd2, reg_p2l.transformation)


