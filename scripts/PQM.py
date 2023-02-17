## Point Quality Metric (PQM) - A metric to measure the difference between  large pointcloud maps

import open3d as o3d
import numpy as np
import sys,os
import copy

# TODO remove deprecated functions

# TODO validiy functions can be combined for better performance

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

def normalizedChamferDistance(pcdA,pcdB,e):
    # pcdA -> ref
    # pcdB -> cand
    # Find the nearest neighbor in pcdB for each point in pcdA
    # Find the nearest neighbor in pcdA for each point in pcdB
    # Calculate the average of the distances
    # Normalize by the average of the point cloud sizes
    # Return the normalized distance
    pcdA = np.asarray(pcdA.points)
    pcdB = np.asarray(pcdB.points)
    treeA = KDTree(pcdA)
    treeB = KDTree(pcdB)
    # Find the nearest neighbor in pcdB for each point in pcdA
    distA, indA = treeA.query(pcdB, k=1)

    # Find valid points based on threshold e
    validA = distA[distA<e]
    # print (len(distA))
    # print (len(validA))
    # Find the nearest neighbor in pcdA for each point in pcdB
    distB, indB = treeB.query(pcdA, k=1)
    # Calculate the average of the distances
    avgDist = (distA.sum() + distB.sum()) / (len(pcdA) + len(pcdB))
    # avgDist = validA.sum() / len(pcdB)
    # Normalize by the average of the point cloud sizes
    # normDist = avgDist / e
    normDist = avgDist / ((len(pcdA) + len(pcdB)) / 2)
    return (1-normDist)

def validityAccuracy(pcdA,pcdB,e):
    # pcdA -> ref
    # pcdB -> cand
    # Normalized Accuracy
    pcdA = np.asarray(pcdA.points)
    pcdB = np.asarray(pcdB.points)
    treeA = KDTree(pcdA)
    # Find the nearest neighbor in pcdB for each point in pcdA
    distA, indA = treeA.query(pcdB, k=1)
    # Find valid points based on threshold e
    validA = distA[distA<e]
    # Normalize by the average of the point cloud sizes
    normAcc = (1-(validA.sum() / len(pcdB)) / e)
    return (normAcc)

# def validityComp(pcdA,pcdB,e):
#     # pcdA -> ref
#     # pcdB -> cand
#     # Normalized completeness
#     pcdA = np.asarray(pcdA.points)
#     pcdB = np.asarray(pcdB.points)
#     treeA = KDTree(pcdA)
#     treeB = KDTree(pcdB)
#     # Find the nearest neighbor in pcdA for each point in pcdB
#     distB, indA = treeB.query(pcdA, k=1)  # FIXME: should be treeA
#     # Find valid points based on threshold e
#     validB = distB[distB<e]
#     completeness = len(validB)/len(pcdA)
#     return (completeness)

# def validityArt(pcdA,pcdB,e):
#     # pcdA -> ref
#     # pcdB -> cand
#     # Normalized completeness
#     pcdA = np.asarray(pcdA.points)
#     pcdB = np.asarray(pcdB.points)
#     treeA = KDTree(pcdA)
#     treeB = KDTree(pcdB)
#     # Find the nearest neighbor in pcdA for each point in pcdB
#     distB, indA = treeA.query(pcdB, k=1)
#     # Find valid points based on threshold e
#     validB = distB[distB<e]
#     artscore =  (len(pcdB) -  len(validB)    /len(pcdB)
#     return (artscore)

def calculateBvalid(pcdA,pcdB,e):
    # pcdA -> ref
    # pcdB -> cand
    # Normalized completeness
    pcdA = np.asarray(pcdA.points)
    pcdB = np.asarray(pcdB.points)
    treeA = KDTree(pcdA)
    treeB = KDTree(pcdB)
    # Find the nearest neighbor in pcdA for each point in pcdB
    distB, indB = treeA.query(pcdB, k=1)
    # Find valid points based on threshold e
    validB = distB[distB<e]
    return (len(validB))

def validityComp(pcdA,pcdB,e):
    # pcdA -> ref
    # pcdB -> cand
    # Normalized completeness
    validB = calculateBvalid(pcdA,pcdB,e)
    return (validB/len(np.asarray(pcdA.points)))

def validityArt(pcdA,pcdB,e):
    # pcdA -> ref
    # pcdB -> cand
    # Normalized artifact score
    validB = calculateBvalid(pcdA,pcdB,e)
    return (validB/len(np.asarray(pcdB.points)))

# def validityArt(pcdA,pcdB,e):
#     # pcdA -> ref
#     # pcdB -> cand
#     # Normalized completeness
#     pcdA = np.asarray(pcdA.points)
#     pcdB = np.asarray(pcdB.points)
#     treeA = KDTree(pcdA)
#     treeB = KDTree(pcdB)
#     # Find the nearest neighbor in pcdA for each point in pcdB
#     distA, indA = treeA.query(pcdB, k=1)
#     # Find valid points based on threshold e
#     validA = distA[distA<e]
#     negart = len(validA)/len(pcdB)
#     return (negart)

# def validityArt(pcdA,pcdB,e):
#     # pcdA -> ref
#     # pcdB -> cand
#     # Normalized 1-artifacts
#     pcdA = np.asarray(pcdA.points)
#     pcdB = np.asarray(pcdB.points)
#     treeA = KDTree(pcdA)
#     # Find the nearest neighbor in pcdA for each point in pcdB
#     distB, indA = treeA.query(pcdA, k=1)
#     # Find valid points based on threshold e
#     validB = distB[distB<e]
#     negartifacts = ((1-len(validB))/len(pcdA))
#     return (negartifacts)

def rmseAccuracy(pcdA,pcdB,e):
    # pcdA -> ref
    # pcdB -> cand
    # Find the nearest neighbor in pcdB for each point in pcdA
    # Find the nearest neighbor in pcdA for each point in pcdB
    # Calculate the average of the distances
    # Normalize by the average of the point cloud sizes
    # Return the normalized distance
    pcdA = np.asarray(pcdA.points)
    pcdB = np.asarray(pcdB.points)
    treeA = KDTree(pcdA)
    treeB = KDTree(pcdB)
    # Find the nearest neighbor in pcdB for each point in pcdA
    distA, indA = treeA.query(pcdB, k=1)
    rmseDist = np.sqrt(np.mean(distA**2)) / (np.mean(distA)+ 1e-6)
    # Find the nearest neighbor in pcdA for each point in pcdB
    # distB, indB = treeB.query(pcdA, k=1)
    # Calculate the average of the distances
    # avgDist = (distA.sum() + distB.sum()) / (len(pcdA) + len(pcdB))
    return (1-np.exp(rmseDist))

def mapQuality(incomp, art, accr, res):
    wIncomp = 0.1
    wArt = 0.1
    wAccr = 0.4
    wRes = 0.4
    # return (res * (accr - (wArt*(art - incomp))))
    # return (1 - (incomp*art*(1-accr)*(1-res))**1/4)
    # Return weighted sum
    return (wIncomp*(1-incomp) + wArt*(1-art) + wAccr*accr + wRes*res)

def validityQuality(comp, negart, accr, res):
    wComp = 0.1
    wNegrt = 0.1
    wAccr = 0.4
    wRes = 0.4
    # return (res * (accr - (wArt*(art - incomp))))
    # return (1 - (incomp*art*(1-accr)*(1-res))**1/4)
    # Return weighted sum
    return (wComp*(comp) + wNegrt*(negart) + wAccr*accr + wRes*res)

def resolutionRatio(pointcloud1, pointcloud2):
    # Return ratio of number of points in pointcloud1 to pointcloud2
    resolution = len(pointcloud1.points) / len(pointcloud2.points)
    if resolution > 1:
        resolution = 1  
    return resolution


def sanityValidComp(pcdA,pcdB,e):
    #Function to find points in pcdB that a valid neighbor in a pcdA based on threshold e
    # pcdA -> ref
    # pcdB -> cand
    pcdA = np.asarray(pcdA.points)
    pcdB = np.asarray(pcdB.points)
    treeA = KDTree(pcdA)
    # Find the nearest neighbor in pcdA for each point in pcdB
    # Here distA means the distance from pcdB to pcdA
    distA, indA = treeA.query(pcdB, k=1)
    # Find valid points based on threshold e
    validA = distA[distA<e]
    # Return the ratio of valid points to total points if ration is less than 1
    if len(validA)/len(pcdB) > 1:
        return 1
    return (len(validA)/len(pcdA))

def sanityValidArt(pcdA,pcdB,e):
    #Function to find points in pcdB that a valid neighbor in a pcdA based on threshold e
    # pcdA -> ref
    # pcdB -> cand
    pcdA = np.asarray(pcdA.points)
    pcdB = np.asarray(pcdB.points)
    treeA = KDTree(pcdA)
    # Find the nearest neighbor in pcdA for each point in pcdB
    # Here distA means the distance from pcdB to pcdA
    distA, indA = treeA.query(pcdB, k=1)
    # Find valid points based on threshold e
    validA = distA[distA<e]
    # Return the ratio of valid points to total points
    return (len(validA)/len(pcdB))

def densityRatio(pointcloud1, pointcloud2):
    # Pointcloud1 -> ref
    # Pointcloud2 -> cand
    # Calculate the density of the pointclouds
    volume1 = np.prod((pointcloud1.get_max_bound() - pointcloud1.get_min_bound()))
    volume2 = np.prod((pointcloud2.get_max_bound() - pointcloud2.get_min_bound()))
    if volume1 != 0:
        density1 = len(pointcloud1.points) / volume1
    else:
        density1 = 0
    if volume2 != 0:
        density2 = len(pointcloud2.points) / volume2
    else:
        density2 = 0
    # Return ratio of density of pointcloud1 to pointcloud2
    # FIXME Cases where density1 or density2 is 0
    if density2 != 0:
        resRatio = density1 / density2
    elif density1 == density2:
        resRatio = 1
    else:
        resRatio = 0
    if resRatio > 1:
        resRatio = 1
    return resRatio


def resolution(pointcloud, MPD,size):
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
    resolution = num_points / (MPD*(size**3))
    
    # If resolution is higher than expected (MPD) then set it to 1.
    if resolution > 1:
        resolution = 1  
    return resolution

if __name__=="__main__":
    pointcloud = o3d.io.read_point_cloud(sys.argv[1])
    pointcloud2 = o3d.io.read_point_cloud(sys.argv[2])

    accr = accuracy(pointcloud,pointcloud2,float(sys.argv[3]))
    res = resolution(pointcloud,100)
    print (accr)
    print (res)