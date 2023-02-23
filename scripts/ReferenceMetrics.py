## This contains metrics to compare our metric against
## Metrics include
## 1. Chamfer Distance
## 2. Hausdorff Distance
## 3. Normalalized Chamfer Distance

from scipy.spatial import distance, KDTree
import numpy as np
import open3d as o3d
from scipy.spatial.distance import cdist
from scipy.spatial.distance import directed_hausdorff
from scipy.stats import wasserstein_distance
import sys,os
import argparse



def calculate_chamfer_distance_metric(pcdA,pcdB):
    """
    Computes the Chamfer Distance between two point clouds
    """
    # Convert point clouds to numpy arrays
    # pcdAnp = np.asarray(pcdA.points)
    # pcdBnp = np.asarray(pcdB.points)

    pcdA = np.asarray(pcdA.points)
    pcdB = np.asarray(pcdB.points)
    treeA = KDTree(pcdA)
    treeB = KDTree(pcdB)
    # Find the nearest neighbor in pcdB for each point in pcdA
    distA, indA = treeB.query(pcdA, k=1)
    # Find the nearest neighbor in pcdA for each point in pcdB
    distB, indB = treeA.query(pcdB, k=1)
    
    # # Compute the average of the minimum distances
    chamfer_dist = distA.sum() + distB.sum()
    
    return chamfer_dist

def calculate_normalized_chamfer_distance_metric(pcdA,pcdB, chamfer_dist):
    # pcdA -> ref
    # pcdB -> cand
    #chamfer_dist = calculate_chamfer_distance_metric(pcdA,pcdB)
    normDist = (2* chamfer_dist )/ ((len(pcdA.points) + len(pcdB.points))**2)
    return (normDist)

def calculate_hausdorff_distance_metric(pcdA,pcdB):
    """
    Computes the Hausdorff Distance between two point clouds
    """
    # Compute the directed Hausdorff distance from pcdA to pcdB
    hausdorff_dist = directed_hausdorff(np.asarray(pcdA.points), np.asarray(pcdB.points))[0]
    
    return hausdorff_dist



def main():
    parser = argparse.ArgumentParser(description='Compute Reference Metrics')
    parser.add_argument('pcdA', help='Path to the reference point cloud')
    parser.add_argument('pcdB', help='Path to the candidate point cloud')
    parser.add_argument('--metric', help='Metric to compute', choices=['chamfer', 'hausdorff', 'normalized_chamfer'], default='chamfer')
    args = parser.parse_args()
    
    # Load the two point clouds
    pcdA = o3d.io.read_point_cloud(args.pcdA)
    pcdB = o3d.io.read_point_cloud(args.pcdB)
    
    # Compute the Chamfer Distance
    if args.metric == 'chamfer':
        chamfer_dist = calculate_chamfer_distance_metric(pcdA, pcdB)
        print("Chamfer Distance: ", chamfer_dist)
    
    # Compute the Normalized Chamfer Distance
    if args.metric == 'normalized_chamfer':
        norm_chamfer_dist = calculate_normalized_chamfer_distance_metric(pcdA, pcdB)
        print("Normalized Chamfer Distance: ", norm_chamfer_dist)
    
    # Compute the Hausdorff Distance
    if args.metric == 'hausdorff':
        hausdorff_dist = calculate_hausdorff_distance_metric(pcdA, pcdB)
        print("Hausdorff Distance: ", hausdorff_dist)


# Examples of how to use this script
# python ReferenceMetrics.py ../data/pcd/000000.pcd ../data/pcd/000001.pcd --metric chamfer
# python ReferenceMetrics.py ../data/pcd/000000.pcd ../data/pcd/000001.pcd --metric hausdorff
# python ReferenceMetrics.py ../data/pcd/000000.pcd ../data/pcd/000001.pcd --metric normalized_chamfer


if __name__ == "__main__":
    main()