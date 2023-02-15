import numpy as np
import open3d as o3d

import copy
import json
#import open3d.pipelines.registration as treg

from util import draw_registration_result, apply_noise, visualize_registered_point_cloud, get_cropping_bound, get_cropped_point_cloud, generate_noisy_point_cloud, generate_grid_lines

from PQM import incompleteness, artifacts, accuracy, resolution

metric_name_to_function = {
    "incompleteness": incompleteness,
    "artifacts": artifacts,
    "accuracy": accuracy,
    "resolution": resolution,
    }


class MapCell:
    def __init__(self, cell_index, pointcloud_gt, pointcloud_cnd, options):
        self.cell_index = cell_index
        self.pointcloud_gt = pointcloud_gt
        self.pointcloud_cnd = pointcloud_cnd
        self.metrics = {}
        self.options = options


    def visualize(self):
        self.pointcloud_gt.paint_uniform_color([1, 0, 0])
        self.pointcloud_cnd.paint_uniform_color([0, 1, 0])
        o3d.visualization.draw_geometries([self.pointcloud_gt, self.pointcloud_cnd])


    def compute_metric(self):
        #compute incompleteness 
        self.metrics["incompleteness"] =metric_name_to_function["incompleteness"](self.pointcloud_gt, self.pointcloud_cnd)
        self.metrics["artifacts"] = metric_name_to_function["artifacts"](self.pointcloud_gt, self.pointcloud_cnd)
        if not self.pointcloud_gt.is_empty() and not self.pointcloud_cnd.is_empty(): 
            #TODO : FIX accuracy computation and then uncomment this
            self.metrics["accuracy"] = "FIX_IT"
            #self.metrics["accuracy"] = metric_name_to_function["accuracy"](self.pointcloud_gt, self.pointcloud_cnd, self.options["e"])
            self.metrics["resolution"] = metric_name_to_function["resolution"](self.pointcloud_cnd, self.options["MPD"])
        else:
            self.metrics["accuracy"] = 0
            self.metrics["resolution"] = 0


class MapMetricManager:
    def __init__(self, pointcloud_GT, pointcloud_Cnd, chunk_size, metric_options = {"e": 0.1, "MPD": 100}):
        self.pointcloud_GT = pointcloud_GT
        self.pointcloud_Cnd = pointcloud_Cnd
        self.chunk_size = chunk_size
        #compute the min bound of the pointcloud
        bb1 = self.pointcloud_Cnd.get_axis_aligned_bounding_box()
        bb2 = self.pointcloud_GT.get_axis_aligned_bounding_box()

        self.min_bound = np.minimum(bb1.min_bound, bb2.min_bound)
        self.max_bound = np.maximum(bb1.max_bound, bb2.max_bound)

        print(self.min_bound, self.max_bound)

        self.cell_dim = (np.ceil((self.max_bound - self.min_bound) / self.chunk_size)).astype(int)
        self.max_bound = self.min_bound + self.cell_dim * self.chunk_size

        print("Dimension of each cell: ", self.cell_dim)

        self.metriccells = {}

        self.options = metric_options


    #visualize pointcloud
    def visualize(self):
        #visualize the pointcloud
        self.pointcloud_GT.paint_uniform_color([1, 0, 0])
        self.pointcloud_Cnd.paint_uniform_color([0, 1, 0])

        o3d.visualization.draw_geometries([self.pointcloud_GT, self.pointcloud_Cnd])


    #visualize pointcloud with grid
    def visualize_cropped_point_cloud(self, chunk_size, min_cell_index, max_cell_index, save=False, filename="test.pcd"):
        #visualize the pointcloud

        cropped_gt, _ = get_cropped_point_cloud(self.pointcloud_GT, self.min_bound, chunk_size, min_cell_index, max_cell_index)
        cropped_gt.paint_uniform_color([1, 0, 0])
        cropped_candidate, _ = get_cropped_point_cloud(self.pointcloud_Cnd, self.min_bound, chunk_size, min_cell_index, max_cell_index)
        #cropped_candidate = self.pointcloud_Cnd.crop(bbox)
        cropped_candidate.paint_uniform_color([0, 1, 0])
        
        if save:
            o3d.io.write_point_cloud(filename+"_gt.pcd", cropped_gt)
            o3d.io.write_point_cloud(filename+"_cnd.pcd", cropped_candidate)
        o3d.visualization.draw_geometries([cropped_gt, cropped_candidate])


    def iterate_cells(self):
        #iterate through all the Cells
        for i in range(int(self.cell_dim[0])):
            for j in range(int(self.cell_dim[1])):
                for k in range(int(self.cell_dim[2])):
                    min_cell_index = np.array([i, j, k])
                    max_cell_index = np.array([i+1, j+1, k+1])
                    #print(min_cell_index, max_cell_index)
                    yield min_cell_index, max_cell_index
  
    def print_points_per_cell(self):
        pcd_list = []
        for min_cell_index, max_cell_index in self.iterate_cells():
            
            cropped_gt, _ = get_cropped_point_cloud(self.pointcloud_GT, self.min_bound, self.chunk_size, min_cell_index, max_cell_index)
            cropped_candidate, _ = get_cropped_point_cloud(self.pointcloud_Cnd, self.min_bound, self.chunk_size, min_cell_index, max_cell_index)
            if cropped_gt.is_empty() or cropped_candidate.is_empty():
                #print("CELL: ", min_cell_index, max_cell_index, end="\t" )
                #print("EMPTY")
                pass
            else:
                #print()                
                pcd_list.append(cropped_gt)
                pcd_list.append(cropped_candidate)


        grid_lines = generate_grid_lines(self.min_bound, self.max_bound, self.cell_dim)

        for line in grid_lines:
            pcd_list.append(line)

        o3d.visualization.draw_geometries(pcd_list)

    def compute_metric(self, filename="test.json"):
        #iterate through all the Cells
        metric_results = {}
        for min_cell_index, max_cell_index in self.iterate_cells():
            
            cropped_gt, _ = get_cropped_point_cloud(self.pointcloud_GT, self.min_bound, self.chunk_size, min_cell_index, max_cell_index)
            cropped_candidate, _ = get_cropped_point_cloud(self.pointcloud_Cnd, self.min_bound, self.chunk_size, min_cell_index, max_cell_index)
            if cropped_gt.is_empty() and cropped_candidate.is_empty():
                pass
            else:
                self.metriccells[str(min_cell_index)] =  MapCell(min_cell_index,cropped_gt, cropped_candidate, self.options)
                self.metriccells[str(min_cell_index)].compute_metric()
                print(self.metriccells[str(min_cell_index)].metrics)
                metric_results[str(min_cell_index)] = self.metriccells[str(min_cell_index)].metrics
                                                                 
        with open(filename, 'w') as fp:
            json.dump(metric_results, fp, indent=4)
            
import sys
import argparse

def main():

    # Example usage
    # argument argparse following variables
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=str, help="GT point cloud")
    parser.add_argument("--cnd", type=str, help="Cnd point cloud")
    parser.add_argument("--sub_sample", help="sub sample size", action="store_true")
    parser.add_argument("--save", help="save file name", action="store_true")
    parser.add_argument("--filename", type=str, help="file name" , default="test_subsample")
    parser.add_argument("--size", type=int, help="sub sample size", default=10)

    parser.add_argument("--min_cell", type=str, help="Min cell index i.e. 0,0,0 or higher", default="2,2,0")
    parser.add_argument("--max_cell", type=str, help="Max cell index i.e. 1,1,1 or higher",default="4,4,1")

    parser.add_argument("--register", help="register point cloud", action="store_true")

    parser.add_argument("--gen_model", help="sub sample size", action="store_true")
    parser.add_argument("--sigma", help="noise sigma ", type=float, default=0.01)

    parser.add_argument("--print", help="print", action="store_true")
    args = parser.parse_args()
    

    min_cell = [int(item) for item in args.min_cell.split(',')]
    max_cell = [int(item) for item in args.max_cell.split(',')]

    pointcloud = o3d.io.read_point_cloud(args.gt)
    pointcloud2 = o3d.io.read_point_cloud(args.cnd)

    mapManager = MapMetricManager(pointcloud,pointcloud2, args.size)

    if args.print:
        mapManager.print_points_per_cell()
    elif args.register:
        visualize_registered_point_cloud(pointcloud,pointcloud2)
    elif args.gen_model:
        generate_noisy_point_cloud(pointcloud, args.sigma, filename=args.filename)
    elif args.sub_sample:
        mapManager.visualize_cropped_point_cloud(args.size, min_cell, max_cell, save=args.save, filename=args.filename)
    else:
        mapManager.visualize()
        


#main entry point
if __name__ == "__main__":

    main()