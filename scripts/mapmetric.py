
from preprocess import MapMetricManager

import open3d as o3d
import numpy as np
import tqdm
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


    parser.add_argument("--weights", type=str, help="4 weights in format [wi, wart, wacc, wr]", default="[0.1, 0.1, 0.4, 0.4]")

    args = parser.parse_args()
    
    weights = eval(args.weights)

    metric_options = {"wi":weights[0], "wart":weights[1], "wacc":weights[2],"wr":weights[3], "e": 0.1}
    mapManager = MapMetricManager(args.gt, args.cnd, args.size, metric_options=metric_options)

    if args.compute:
        with tqdm.tqdm(total=100) as pbar:
            mapManager.compute_metric(args.filename)
            pbar.update(100)
        # mapManager.compute_metric(args.filename)
        
    elif args.print:
        mapManager.print_points_per_cell()
    else:
        #draw_registration_result(pointcloud,pointcloud2, np.eye(4))
        mapManager.visualize()
        


#main entry point
if __name__ == "__main__":

    main()