

import numpy as np
from sklearn.neighbors import KDTree



def calculate_average_distance(pcdA):
    # pcdA -> ref
    # Calculate the average distance between points in pcdA
    pcdA = np.asarray(pcdA.points)
    treeA = KDTree(pcdA)
    # Find the nearest neighbor in pcdA for each point in pcdA
    distA, indA = treeA.query(pcdA, k=2)
    # Calculate the average distance between points in pcdA
    avgDistA = np.mean(distA[:,1])
    return avgDistA

def calculate_Bvalid_dist(pcdA,pcdB,e):
    # pcdA -> ref
    # pcdB -> cand
    # distances of valid points in pcdB
    pcdA = np.asarray(pcdA.points)
    pcdB = np.asarray(pcdB.points)
    treeA = KDTree(pcdA)
    treeB = KDTree(pcdB)
    # Find the nearest neighbor in pcdA for each point in pcdB
    distB, indB = treeA.query(pcdB, k=1)
    # Find valid points based on threshold e
    validB = distB[distB<e]
    return validB

def calculate_Bvalid_count(pcdA,pcdB,e):
    # pcdA -> ref
    # pcdB -> cand
    # Normalized completeness
    validB_dist = calculate_Bvalid_dist(pcdA,pcdB,e)
    return len(validB_dist)


def calculate_completeness_metric(pcdA,pcdB,e):
    # pcdA -> ref
    # pcdB -> cand
    # Normalized completeness
    validB_count = calculate_Bvalid_count(pcdA,pcdB,e)
    normCompleteness = (validB_count/len(np.asarray(pcdA.points)))
    if normCompleteness > 1.0:
        return 1.0
    return normCompleteness

def calculate_artifacts_metric(pcdA,pcdB,e):
    # pcdA -> ref
    # pcdB -> cand
    # Normalized artifact score
    validB_count = calculate_Bvalid_count(pcdA,pcdB,e)
    return (validB_count/len(np.asarray(pcdB.points)))


def calculate_volume(pcd):
    # Calculate the volume of the pointcloud
    volume = np.prod((pcd.get_max_bound() - pcd.get_min_bound()))
    return volume

def calculate_density(pcd):
    # Calculate the density of the pointcloud
    volume = calculate_volume(pcd)
    if volume != 0:
        density = len(pcd.points) / volume
    else:
        density = 0
    return density

def calculate_resolution_metric(pcdA,pcdB):
    # pcdA -> ref
    # pcdB -> cand
    # Normalized resolution score
    # Calculate the density of the pointclouds
    densityA = calculate_density(pcdA)
    densityB = calculate_density(pcdB)
    # Return ratio of density of pcdA to pcdB
    # FIXME Cases where densityA or densityB is 0
    if densityB != 0:
        resRatio = densityA / densityB
    elif densityA == densityB:
        resRatio = 1.0
    else:
        resRatio = 0
    if resRatio > 1:
        resRatio = 1
    return resRatio


def calculate_accuracy_metric(pcdA,pcdB,e):
    # pcdA -> ref
    # pcdB -> cand
    # Normalized accuracy score
    # Find the nearest neighbor in pcdB for each point in pcdA
    validB_dist = calculate_Bvalid_dist(pcdA,pcdB,e)

    normAcc = 1-(validB_dist.sum() / (len(pcdB.points) * e))
    return normAcc


def calculate_complete_quality_metric_old(pcdA,pcdB, e, wc, wt,wr, wa):
    # pcdA -> ref
    # pcdB -> cand
    # Normalized complete quality score
    # Calculate the completeness metric
    normComp = calculate_completeness_metric(pcdA,pcdB,e)
    # Calculate the artifacts metric
    normArt = calculate_artifacts_metric(pcdA,pcdB,e)
    # Calculate the resolution metric
    normRes = calculate_resolution_metric(pcdA,pcdB)
    # Calculate the accuracy metric
    normAcc = calculate_accuracy_metric(pcdA,pcdB,e)
    # Calculate the complete quality metric
    normCompQual = (wc*normComp) + (wt*normArt) + (wr*normRes) + (wa*normAcc)
    return normCompQual , normComp, normArt, normRes, normAcc

###############################################################################


def calculate_completeness_metric_fast(validB_count, A_Count):
    # pcdA -> ref
    # pcdB -> cand
    # Normalized completeness
    normCompleteness = (validB_count/A_Count)
    if normCompleteness > 1.0:
        return 1.0
    return normCompleteness



def calculate_artifacts_metric_fast(validB_count, B_Count):
    # pcdA -> ref
    # pcdB -> cand
    # Normalized artifact score
    return (validB_count/B_Count)


def calculate_accuracy_metric_fast(validB_dist, B_Count, e):
    # pcdA -> ref
    # pcdB -> cand
    # Normalized accuracy score
    # Find the nearest neighbor in pcdB for each point in pcdA

    normAcc = 1-(validB_dist.sum() / (B_Count * e))
    return normAcc


def calculate_complete_quality_metric(pcdA,pcdB, e, wc, wt,wr, wa): 
    # pcdA -> ref
    # pcdB -> cand
    # Normalized complete quality score

    validB_dist = calculate_Bvalid_dist(pcdA,pcdB,e)
    validB_count = len(validB_dist)

    A_Count = len(np.asarray(pcdA.points))
    B_Count = len(np.asarray(pcdB.points))
    
    # Calculate the completeness metric
    normComp = calculate_completeness_metric_fast(validB_count, A_Count)
    # Calculate the artifacts metric
    normArt = calculate_artifacts_metric_fast(validB_count, B_Count)
    # Calculate the resolution metric
    normRes = calculate_resolution_metric(pcdA,pcdB)
    # Calculate the accuracy metric
    normAcc = calculate_accuracy_metric_fast(validB_dist, B_Count, e)
    # Calculate the complete quality metric
    normCompQual = (wc*normComp) + (wt*normArt) + (wr*normRes) + (wa*normAcc)
    return normCompQual , normComp, normArt, normRes, normAcc
