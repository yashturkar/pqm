## Point Quality Metric (PQM) - A metric to measure the difference between  large pointcloud maps

import open3d as o3d
import numpy as np
import sys,os
import copy

## Not sure if this works the way based on counting
# def incompleteness(pcd_gt, pcd_cand):
#     """
#     Calculates the incompleteness score of pcd_cand with respect to pcd_gt.

#     Parameters:
#     - pcd_gt: open3d.geometry.PointCloud, the ground truth point cloud
#     - pcd_cand: open3d.geometry.PointCloud, the candidate point cloud

#     Returns:
#     - incompleteness: float, the incompleteness score
#     """
#     # Convert point clouds to numpy arrays
#     gt_points = pcd_gt.points
#     cand_points = pcd_cand.points

#     # Calculate the number of points in gt but not in cand
#     diff_points = o3d.geometry.PointCloud()
#     diff_points.points = o3d.utility.Vector3dVector(gt_points)
#     diff_points.paint_uniform_color([1, 0, 0])  # Paint ground truth points in red
#     diff_points.remove_points(cand_points)
#     num_diff_points = len(diff_points.points)

#     # Calculate the incompleteness score
#     num_gt_points = len(gt_points)
#     incompleteness = num_diff_points / num_gt_points

#     return incompleteness


def incompleteness(chunkA, chunkB):
    # Using only number of points, when region or chunk size tends to 0, point-point works out (hopefully)
    # chunk a -> ref
    # chunk b -> cand
    numchunkA = len(chunkA.points)
    numchunkB = len(chunkB.points)

    if numchunkA >= numchunkB:
        return ((numchunkA-numchunkB)/numchunkA)
    return 0
    
def artifacts(chunkA, chunkB):
    # Using only number of points, when region or chunk size tends to 0, point-point works out (hopefully)
    # chunkA -> ref
    # chunkB -> cand
    numchunkA = len(chunkA.points)
    numchunkB = len(chunkB.points)

    if numchunkB >= numchunkA:
        return ((numchunkB-numchunkA)/numchunkB)
    return 0


def totalIncompleteness(chunksA,chunksB):
    # chunksA -> list of chunks of ref
    # chunksB -> list of chunks of cand
    totalIncomp = 0
    for chunkA, chunkB in zip(chunksA,chunksB):
        totalIncomp += incompleteness(chunkA,chunkB)
    return totalIncomp

def totalArtifacts(chunksA,chunksB):
    # chunksA -> list of chunks of ref
    # chunksB -> list of chunks of cand
    totalArt = 0
    for chunkA, chunkB in zip(chunksA,chunksB):
        totalArt += artifacts(chunkA,chunkB)
    return totalArt


# TODO Gen test
def accuracy(GT, Cand, e):
    """
    Finds the nearest neighbor in GT for each point in Cand and counts the number of matches
    where the distance between the point and its nearest neighbor is less than e.

    Parameters:
    -----------
    GT : open3d.geometry.PointCloud
        The ground truth point cloud.
    Cand : open3d.geometry.PointCloud
        The candidate point cloud.
    e : float
        The maximum distance between a point in Cand and its nearest neighbor in GT for the match to be counted.

    Returns:
    --------
    num_matches : int
        The number of matches where the distance between the point and its nearest neighbor is less than e.
    num_mismatches : int
        The number of mismatches where the distance between the point and its nearest neighbor is greater than or equal to e.
    """
    num_matches = 0
    num_mismatches = 0

    # Convert the open3d point clouds to numpy arrays
    GT_np = np.asarray(GT.points)
    Cand_np = np.asarray(Cand.points)

    # Create a copy of GT to avoid modifying the original array
    GT_copy = copy.deepcopy(GT_np)

    # Loop over each point in Cand
    for cand_point in Cand_np:
        # Calculate the distances between the candidate point and all points in GT
        distances = np.linalg.norm(GT_copy - cand_point, axis=1)

        # Find the index of the nearest neighbor
        nn_index = np.argmin(distances)

        # Get the distance to the nearest neighbor
        nn_distance = distances[nn_index]

        # If the distance is less than e, increment the match counter
        if nn_distance < e:
            num_matches += 1
        else:
            num_mismatches += 1

        # Remove the nearest neighbor from GT so it is not used again
        GT_copy = np.delete(GT_copy, nn_index, axis=0)
        if len(GT_copy) == 0:
            break
    accr = num_matches / (num_matches + num_mismatches)
    return accr

    #return num_matches, num_mismatches


from scipy.spatial import distance, KDTree


def accuracy_fast(GT, Cand, e):
    # find 1 nearest neighbor in GT for each point in Cand
    
    # Convert the open3d point clouds to numpy arrays
    GT_np = np.asarray(GT.points)
    Cand_np = np.asarray(Cand.points)

    # Create a copy of GT to avoid modifying the original array
    GT_copy = copy.deepcopy(GT_np)

    dm = distance.cdist(GT_np, Cand_np, 'euclidean')
    ix = np.argmin(dm, 0)
    min_dist = np.min(dm, 0)
    #print(GT_np.shape, Cand_np.shape, ix.shape)
    #print(dm.shape, ix[0], min_dist[0])
    num_matches = (min_dist < e).sum() 
    num_mismatches = (min_dist >= e).sum()

    #print(ix.shape, np.unique(ix).shape)
    #num_matches1, num_mismatches1 = accuracy(GT, Cand, e)
    #print(num_matches, num_mismatches, num_matches1, num_mismatches1)
    accr = num_matches / (num_matches + num_mismatches)
    return accr


def resolution(pointcloud, MPD):
    """
    Calculates the resolution of a pointcloud by dividing the number of points by the given MPD.

    Parameters:
        pointcloud (o3d.geometry.PointCloud): The input pointcloud.
        MPD (float): The minimum point distance.

    Returns:
        float: The resolution of the pointcloud.
    """
    # Calculate the number of points in the pointcloud.
    num_points = len(pointcloud.points)

    # Calculate the resolution by dividing the number of points by the MPD.
    resolution = num_points / MPD

    return resolution


# pointcloud = o3d.io.read_point_cloud(sys.argv[1])
# pointcloud2 = o3d.io.read_point_cloud(sys.argv[2])

# accr = accuracy(pointcloud,pointcloud2,float(sys.argv[3]))
# res = resolution(pointcloud,100)
# print (accr)
# print (res)