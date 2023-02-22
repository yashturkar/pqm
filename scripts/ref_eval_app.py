
import argparse
import json
import os
from tqdm import tqdm
import open3d as o3d
import subprocess as sp
from system_constants import *
from multiprocessing import Pool


def calcChamferDistance(gt, cnd):
    cd = sp.check_output(["pdal", "chamfer", gt, cnd])
    cd = cd.decode("utf-8")
    cd = cd.splitlines()
    cd = cd[1].split(":")
    cd = cd[1]
    cd = cd.strip()
    cd = float(cd[:-1])
    return cd

def calcHausdorffDistance(gt, cnd):
    hd = sp.check_output(["pdal", "hausdorff", gt, cnd])
    hd = hd.decode("utf-8")
    hd = hd.splitlines()
    hd = hd[6].split(":")
    hd = hd[1]
    hd = float(hd[:-1])
    return hd

def calcTotalPoints(gt, cnd):
    gt = o3d.io.read_point_cloud(gt)
    cnd = o3d.io.read_point_cloud(cnd)

    total_gt = len(gt.points)
    total_cnd = len(cnd.points)
    return total_gt, total_cnd


def calcNormalizedChamferDistance(gt, cnd, total_gt, total_cnd):
    cd = calcChamferDistance(gt, cnd)
    ncd = (2 * cd) / (total_gt + total_cnd)
    return ncd


def calculate_metrics(args):
    cnd_path, save_path, gt_path = args
    if not os.path.exists(cnd_path):
        print("cnd path not exist")
        return
    # Create json file for each cnd file
    json_path = os.path.join(save_path, os.path.basename(cnd_path[:-4])+"_ref_metrics" + ".json")
    # Create dictionary to store metrics
    metrics = {}
    # Calculate chamfer distance
    cd = calcChamferDistance(gt_path, cnd_path)
    # Calculate hausdorff distance
    hd = calcHausdorffDistance(gt_path, cnd_path)
    # Total points
    total_gt,total_cnd = calcTotalPoints(gt_path, cnd_path)
    # Add chamfer and hausdorff distances to metrics dictionary
    metrics["chamfer"] = cd
    metrics["hausdorff"] = hd
    # Add total number of points to metrics dictionary
    metrics["total_gt"] = total_gt
    metrics["total_cnd"] = total_cnd
    # Add GT and CND file paths to metrics dictionary
    metrics["gt_path"] = gt_path
    metrics["cnd_path"] = cnd_path
    # Calculate normalized chamfer distance
    ncd = calcNormalizedChamferDistance(gt_path, cnd_path, total_gt, total_cnd)
    # Add normalized chamfer distance to metrics dictionary
    metrics["normalized_chamfer"] = ncd
    # Add metrics dictionary to json file in new line
    with open(json_path, mode='w') as fp:
        json.dump(metrics, fp, indent=4)
        fp.write("\n")

def compute_metrics(cnd_paths, save_dir, gt_path, num_processes):
    # Create list of arguments for calculate_metrics function
    args_list = [(cnd_path, save_dir, gt_path) for cnd_path in cnd_paths]
    # Create multiprocessing pool
    with Pool(num_processes) as pool:
        for _ in tqdm(pool.imap_unordered(calculate_metrics, args_list), total=len(args_list), desc="Calculating metrics", unit="file"):
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config json file")
    # Add argument for number of processes
    parser.add_argument("--num_processes", type=int, help="number of processes to use",default=4)
    num_processes = parser.parse_args().num_processes

    args = parser.parse_args()
    config = {}
    with open(args.config, mode='r') as fp:
        config = json.load(fp)

        gt_path = config[CONFIG_GT_FILE_STR]
        cnd_paths = config[CONFIG_CND_FILE_STR]
        save_path = config[CONFIG_SAVE_PATH_STR]

        size = config[CELL_SIZE_STR]
        weights = config[CONFIG_WEIGHTS_STR]
        eps = config[CONFIG_EPS_STR]

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(gt_path):
            print("gt path not exist")
            return
        compute_metrics(cnd_paths, save_path, gt_path, num_processes)




if __name__ == "__main__":
    main()

