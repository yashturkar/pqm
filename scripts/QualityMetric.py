

import numpy as np
from sklearn.neighbors import KDTree


from scipy.spatial import cKDTree
from sklearn.metrics import euclidean_distances
import numpy as np
import multiprocessing as mp

import torch
from torch.utils.data import DataLoader, TensorDataset



def calculate_Bvalid_dist(pcdA, pcdB, e, batch_size=1024):
    # pcdA -> ref
    # pcdB -> cand
    # distances of valid points in pcdB
    pcdA = torch.tensor(np.asarray(pcdA.points))
    pcdB = torch.tensor(np.asarray(pcdB.points))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pcdA = pcdA.to(device)
    pcdB = pcdB.to(device)
    
    with torch.no_grad():
        # Create a dataset from pcdB
        dataset = TensorDataset(pcdB)
        
        # Create a DataLoader to load the data in batches
        loader = DataLoader(dataset, batch_size=batch_size)
        
        # Find the nearest neighbor in pcdA for each point in pcdB
        distB, indB = [], []
        for batch in loader:
            batch = batch[0]
            dists = torch.cdist(pcdA, batch, p=2)
            distB_batch, indB_batch = torch.min(dists, dim=0)
            distB.append(distB_batch.cpu())
            indB.append(indB_batch.cpu())
        distB = torch.cat(distB, dim=0).numpy()
        indB = torch.cat(indB, dim=0).numpy()
        
        # Find valid points based on threshold e
        validB = distB[distB<e]
        
    return validB
def calculate_Bvalid_dist_cuda(pcdA, pcdB, e):
    # pcdA -> ref
    # pcdB -> cand
    # distances of valid points in pcdB
    pcdA = torch.tensor(np.asarray(pcdA.points))
    pcdB = torch.tensor(np.asarray(pcdB.points))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pcdA = pcdA.to(device)
    pcdB = pcdB.to(device)
    
    with torch.no_grad():
        # Compute distances between pcdA and pcdB
        dists = torch.cdist(pcdA, pcdB, p=2)
        
        # Find the nearest neighbor in pcdA for each point in pcdB
        distB, indB = torch.min(dists, dim=0)
        
        # Move the data back to the CPU and convert to numpy arrays
        distB = distB.cpu().numpy()
        
        # Find valid points based on threshold e
        validB = distB[distB<e]
        
    return validB



def calculate_Bvalid_dist_fast(pcdA,pcdB,e, n_jobs=1):
    # pcdA -> ref
    # pcdB -> cand
    # distances of valid points in pcdB
    pcdA = np.asarray(pcdA.points)
    pcdB = np.asarray(pcdB.points)
    
    if n_jobs <= 1:
        treeA = cKDTree(pcdA)
        # Find the nearest neighbor in pcdA for each point in pcdB
        distB, indB = treeA.query(pcdB, k=1)
        # Find valid points based on threshold e
        validB = distB[distB<e]
    else:
        if len(pcdB) < n_jobs:
            n_jobs = len(pcdB)
        chunk_size = int(len(pcdB) / n_jobs)
        chunks = [pcdB[i:i+chunk_size] for i in range(0, len(pcdB), chunk_size)]
        
        pool = mp.Pool(processes=n_jobs)
        results = pool.starmap(_process_chunk, [(pcdA, chunk, e) for chunk in chunks])
        pool.close()
        pool.join()
        
        validB = np.concatenate([r[0] for r in results])

    return validB

def _process_chunk(pcdA, pcdB, e):
    treeA = cKDTree(pcdA)
    # Find the nearest neighbor in pcdA for each point in pcdB
    distB, indB = treeA.query(pcdB, k=1)
    # Find valid points based on threshold e
    validB = distB[distB<e]
    return validB,




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

def calculate_Bvalid_count(pcdA,pcdB,e,validB_dist):
    # pcdA -> ref
    # pcdB -> cand
    # Normalized completeness
    # validB_dist = calculate_Bvalid_dist(pcdA,pcdB,e)
    return len(validB_dist)


def calculate_completeness_metric(pcdA,pcdB,e,validB_count):
    # pcdA -> ref
    # pcdB -> cand
    # Normalized completeness
    # validB_count = calculate_Bvalid_count(pcdA,pcdB,e)
    normCompleteness = (validB_count/len(np.asarray(pcdA.points)))
    if normCompleteness > 1.0:
        return 1.0
    return normCompleteness

def calculate_artifacts_metric(pcdA,pcdB,e,validB_count):
    # pcdA -> ref
    # pcdB -> cand
    # Normalized artifact score
    # validB_count = calculate_Bvalid_count(pcdA,pcdB,e)
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


def calculate_accuracy_metric(pcdA,pcdB,e,validB_dist):
    # pcdA -> ref
    # pcdB -> cand
    # Normalized accuracy score
    # Find the nearest neighbor in pcdB for each point in pcdA
    # validB_dist = calculate_Bvalid_dist(pcdA,pcdB,e)
    normAcc = 1-(validB_dist.sum() / (len(pcdB.points) * e))
    return normAcc


def calculate_complete_quality_metric_old(pcdA,pcdB, e, wc, wt,wr, wa):
    # pcdA -> ref
    # pcdB -> cand
    # Normalized complete quality score
    validB_dist = calculate_Bvalid_dist(pcdA,pcdB,e,batch_size=256)
    validB_count = calculate_Bvalid_count(pcdA,pcdB,e,validB_dist)
    # Calculate the completeness metric
    normComp = calculate_completeness_metric(pcdA,pcdB,e,validB_count)
    # Calculate the artifacts metric
    normArt = calculate_artifacts_metric(pcdA,pcdB,e,validB_count)
    # Calculate the resolution metric
    normRes = calculate_resolution_metric(pcdA,pcdB)
    # Calculate the accuracy metric
    normAcc = calculate_accuracy_metric(pcdA,pcdB,e,validB_dist)
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
