
from preprocess import MapMetricManager

import open3d as o3d
import numpy as np

import argparse

GT_COLOR = [0, 1, 0]
CND_COLOR = [0, 0, 1]

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

    parser.add_argument("--e", type=float, help="e", default=5.0)
    parser.add_argument("--MPD", type=float, help="MPD", default=100)

    args = parser.parse_args()
    


    metric_options = {"e": args.e , "MPD": args.MPD}
    mapManager = MapMetricManager(args.gt, args.cnd, args.size, metric_options=metric_options)

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