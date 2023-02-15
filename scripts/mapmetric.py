
from preprocess import MapMetricManager

import open3d as o3d
import numpy as np

import argparse

def main():

    # Example usage
    # argument argparse following variables
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=str, help="GT point cloud")
    parser.add_argument("--cnd", type=str, help="Cnd point cloud")
    parser.add_argument("--save", help="save file name", action="store_true")
    parser.add_argument("--filename", type=str, help="file name" , default="results/test_metric.json")
    parser.add_argument("--size", type=int, help="sub sample size", default=10)

    parser.add_argument("--print", help="print", action="store_true")
    parser.add_argument("--compute", help="compute", action="store_true")
    args = parser.parse_args()
    

    pointcloud = o3d.io.read_point_cloud(args.gt)
    pointcloud2 = o3d.io.read_point_cloud(args.cnd)

    mapManager = MapMetricManager(pointcloud,pointcloud2, args.size)

    if args.compute:
        mapManager.compute_metric(args.filename)
        
    elif args.print:
        mapManager.print_points_per_cell()
    else:
        #draw_registration_result(pointcloud,pointcloud2, np.eye(4))
        mapManager.visualize()
        


#main entry point
if __name__ == "__main__":

    main()