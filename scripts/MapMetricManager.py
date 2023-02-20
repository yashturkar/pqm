

from QualityMetric import  calculate_complete_quality_metric, calculate_density
import numpy as np
import open3d as o3d
from tqdm import tqdm
import copy
import json
import pickle as pkl
from util import get_cropping_bound, get_cropped_point_cloud, generate_grid_lines, get_cropped_box

from ReferenceMetrics import calculate_chamfer_distance_metric, calculate_normalized_chamfer_distance_metric, calculate_hausdorff_distance_metric

from system_constants import *
class MapCell:
    def __init__(self, cell_index, pointcloud_gt, pointcloud_cnd, options, fill_metrics = True):
        self.cell_index = cell_index
        self.metrics = {}
        self.options = options
        if fill_metrics:        
            if not pointcloud_gt.is_empty() and not pointcloud_cnd.is_empty():
                self.metrics[QUALITY_STR], self.metrics[COMPELTENESS_STR], self.metrics[ARTIFACTS_STR], self.metrics[RESOLUTION_STR], self.metrics[ACCURACY_STR]= calculate_complete_quality_metric(pointcloud_gt, pointcloud_cnd, options[EPSILON_STR], options[WEIGHT_COMPLETENESS_STR], options[WEIGHT_ARTIFACTS_STR], options[WEIGHT_RESOLUTION_STR], options[WEIGHT_ACCURACY_STR])

            else:
                self.metrics[COMPELTENESS_STR] = 0
                self.metrics[ARTIFACTS_STR] = 0
                self.metrics[RESOLUTION_STR] = 0
                self.metrics[ACCURACY_STR] = 0
                self.metrics[QUALITY_STR] = 1.0
            self.metrics[CHAMFER_STR] = calculate_chamfer_distance_metric(pointcloud_gt, pointcloud_cnd)
            self.metrics[NORMALIZED_CHAMFER_STR] = calculate_normalized_chamfer_distance_metric(pointcloud_gt, pointcloud_cnd)
            self.metrics[HOUSDORFF_STR] = calculate_hausdorff_distance_metric(pointcloud_gt, pointcloud_cnd)
                
                                                                         

def parse_mapmetric_cells(cell_index, options, cell_metrics):
    cell = MapCell(cell_index, None, None, options, fill_metrics = False)
    cell.metrics = cell_metrics
    return cell

def get_list_from_string(cell_index):
    cell_indx_val_tmp = cell_index.replace("[", "").replace("]", "").split(" ")
    cell_indx_val = np.array([int(x) for x in cell_indx_val_tmp if x.strip() != ""])
    return cell_indx_val


def parse_mapmetric_config(config_file):
    with open(config_file) as f:
        config = json.load(f)
        map_metric = MapMetricManager(config[CONFIG_GT_FILE_STR], config[CONFIG_CND_FILE_STR], config[CELL_SIZE_STR], config[CONFIG_OPTIONS_STR])
        for cell_index in config[METRICS_STR]:
            # cell_indx_val_tmp = cell_index.replace("[", "").replace("]", "").split(" ")
            # cell_indx_val = np.array([int(x) for x in cell_indx_val_tmp if x.strip() != ""])
            map_metric.metriccells[cell_index] = parse_mapmetric_cells(get_list_from_string(cell_index), config[CONFIG_OPTIONS_STR], config[METRICS_STR][cell_index])
        return map_metric
    


class MapMetricManager:
    def __init__(self, gt_file, cnd_file, cell_size, metric_options = {EPSILON_STR: 0.1, WEIGHT_COMPLETENESS_STR: 0.1, WEIGHT_ARTIFACTS_STR: 0.1, WEIGHT_RESOLUTION_STR: 0.4, WEIGHT_ACCURACY_STR: 0.4}):

        self.gt_file = gt_file
        self.cnd_file = cnd_file

        self.pointcloud_GT = o3d.io.read_point_cloud(gt_file)
        self.pointcloud_Cnd = o3d.io.read_point_cloud(cnd_file)

        self.pointcloud_GT.paint_uniform_color(GT_COLOR)
        self.pointcloud_Cnd.paint_uniform_color(CND_COLOR)

        self.cell_size = cell_size
        metric_options["r"] = cell_size
        #compute the min bound of the pointcloud
        bb1 = self.pointcloud_Cnd.get_axis_aligned_bounding_box()
        bb2 = self.pointcloud_GT.get_axis_aligned_bounding_box()

        self.min_bound = np.minimum(bb1.min_bound, bb2.min_bound)
        self.max_bound = np.maximum(bb1.max_bound, bb2.max_bound)

        #print(self.min_bound, self.max_bound)

        self.cell_dim = (np.ceil((self.max_bound - self.min_bound) / self.cell_size)).astype(int)
        self.max_bound = self.min_bound + self.cell_dim * self.cell_size

        print("Dimension of each cell: ", self.cell_dim)

        self.metriccells = {}

        self.options = metric_options

    def reset(self, cell_size, metric_options = {EPSILON_STR: 0.1, WEIGHT_COMPLETENESS_STR: 0.1, WEIGHT_ARTIFACTS_STR: 0.1, WEIGHT_RESOLUTION_STR: 0.4, WEIGHT_ACCURACY_STR: 0.4}):
        self.metriccells = {}
        self.cell_size = cell_size
        self.cell_dim = (np.ceil((self.max_bound - self.min_bound) / self.cell_size)).astype(int)
        self.max_bound = self.min_bound + self.cell_dim * self.cell_size
        self.options = metric_options
        print("Dimension of each cell: ", self.cell_dim)        

    #visualize pointcloud
    def visualize(self):
        #visualize the pointcloud
        self.pointcloud_GT.paint_uniform_color(RED_COLOR)
        self.pointcloud_Cnd.paint_uniform_color(GREEN_COLOR)

        o3d.visualization.draw_geometries([self.pointcloud_GT, self.pointcloud_Cnd])

    def get_heatmap(self, metric_name):
        heatmap = copy.deepcopy(self.pointcloud_GT) #o3d.geometry.PointCloud()
        heatmap.paint_uniform_color([0, 0, 0])
        colors = np.zeros((len(heatmap.points), 3))
        for cell_index in self.metriccells:
            cell = self.metriccells[cell_index]
            cell_index_vals = get_list_from_string(cell_index)
            cell_index_next = cell_index_vals + np.array([1, 1, 1])

            min_bound, max_bound = get_cropping_bound(self.min_bound, self.cell_size, cell_index_vals, cell_index_next)
            #print(min_bound, max_bound)
            bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)

            col_indx = bbox.get_point_indices_within_bounding_box(heatmap.points)
            
            color = cell.metrics[metric_name] * GREEN_COLOR + (1-cell.metrics[metric_name]) * RED_COLOR

            #colors[col_indx] = [1-cell.metrics[metric_name], cell.metrics[metric_name], 0]
            colors[col_indx] = color
            #cropped_gt.paint_uniform_color([cell.metrics[metric_name], 0, 0])

            #cell_center = self.min_bound + (cell.cell_index + 0.5) * self.cell_size
            #heatmap.points.append(cell_center)
        heatmap.colors = o3d.utility.Vector3dVector(colors) # append([cell.metrics[metric_name], 0, 0])
        return heatmap
    
    #visualize pointcloud heatmap
    def visualize_heatmap(self, metric_name, save=False, filename="test_heatmap.pcd"):
        #visualize the pointcloud
        heatmap = self.get_heatmap(metric_name)
        #heatmap.paint_uniform_color(RED_COLOR)
        if save:
            o3d.io.write_point_cloud(filename, heatmap)
        o3d.visualization.draw_geometries([heatmap])

    #visualize pointcloud with grid
    def visualize_cropped_point_cloud(self, cell_size, min_cell_index, max_cell_index, save=False, filename="test.pcd"):
        #visualize the pointcloud

        cropped_gt, _ = get_cropped_point_cloud(self.pointcloud_GT, self.min_bound, cell_size, min_cell_index, max_cell_index)
        cropped_gt.paint_uniform_color(RED_COLOR)
        cropped_candidate, _ = get_cropped_point_cloud(self.pointcloud_Cnd, self.min_bound, cell_size, min_cell_index, max_cell_index)
        #cropped_candidate = self.pointcloud_Cnd.crop(bbox)
        cropped_candidate.paint_uniform_color(GREEN_COLOR)
        
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
            
            cropped_gt, _ = get_cropped_point_cloud(self.pointcloud_GT, self.min_bound, self.cell_size, min_cell_index, max_cell_index)
            cropped_candidate, _ = get_cropped_point_cloud(self.pointcloud_Cnd, self.min_bound, self.cell_size, min_cell_index, max_cell_index)
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
        numIter = self.cell_dim[0] * self.cell_dim[1] * self.cell_dim[2]
        print("Number of Iterations: ", numIter)
        for i,(min_cell_index, max_cell_index) in tqdm(enumerate(self.iterate_cells()), total=numIter):
        # for min_cell_index, max_cell_index in self.iterate_cells():
            
            cropped_gt,gt_box = get_cropped_point_cloud(self.pointcloud_GT, self.min_bound, self.cell_size, min_cell_index, max_cell_index)
            cropped_candidate,cand_box = get_cropped_point_cloud(self.pointcloud_Cnd, self.min_bound, self.cell_size, min_cell_index, max_cell_index)
            if cropped_gt.is_empty() and cropped_candidate.is_empty():
                pass
            else:

                self.metriccells[str(min_cell_index)] =  MapCell(min_cell_index,cropped_gt, cropped_candidate, self.options)                
                # print(self.metriccells[str(min_cell_index)].metrics)

                                                                 
        self.save_metric(filename)


    def compute_metric_average(self, metric_results):
        #iterate through all the Cells
        metric_results[AVERAGE_STR] = {}
        metric_results[VARIANCE_STR] = {}
        for metric in METRICS_LIST:
            metric_list = [metric_results[METRICS_STR][key][metric] for key in metric_results[METRICS_STR].keys()]
            average_metric = np.mean(metric_list)
            print("Average ", metric, ": ", average_metric)
            metric_var = np.var(metric_list)
            print("Variance for ", metric, ": ", metric_var)
            metric_results[AVERAGE_STR][metric] = average_metric
            metric_results[VARIANCE_STR][metric] = metric_var
                                                                 
        return metric_results


    def save_metric(self, filename="test.json"):
        metric_results = {}   
        metric_results[METRICS_STR]={}
        for key in self.metriccells.keys():
           metric_results[METRICS_STR][key] = self.metriccells[key].metrics

        metric_results[CELL_SIZE_STR] = self.cell_size
        metric_results[CELL_DIM_STR] = self.cell_dim.flatten().tolist()
        metric_results[MIN_BOUND_STR] = self.min_bound.flatten().tolist()
        metric_results[MAX_BOUND_STR] = self.max_bound.flatten().tolist()
        metric_results[CONFIG_OPTIONS_STR] = self.options
        metric_results[CONFIG_GT_FILE_STR] = self.gt_file
        metric_results[CONFIG_CND_FILE_STR] = self.cnd_file

        metric_results = self.compute_metric_average(metric_results)


        metric_results[DENSITY_GT_STR] = calculate_density(self.pointcloud_GT)
        metric_results[DENSITY_CND_STR] = calculate_density(self.pointcloud_Cnd)

        metric_results[CHAMFER_STR]=  calculate_chamfer_distance_metric(self.pointcloud_GT, self.pointcloud_Cnd)
        metric_results[NORMALIZED_CHAMFER_STR]=  calculate_normalized_chamfer_distance_metric(self.pointcloud_GT, self.pointcloud_Cnd)
        metric_results[HOUSDORFF_STR]=  calculate_hausdorff_distance_metric(self.pointcloud_GT, self.pointcloud_Cnd)
        print("================Summary======================")
        print("Our Metric: ", metric_results[AVERAGE_STR][QUALITY_STR])
        print("Chamfer Metric: ", metric_results[CHAMFER_STR])
        print("Normalized Chamfer Metric: ", metric_results[NORMALIZED_CHAMFER_STR])
        print("Hausdorff Metric: ", metric_results[HOUSDORFF_STR])


        with open(filename, 'w+') as fp:
            json.dump(metric_results, fp, indent=4)


    def compute_metric_fast(self, filename ="test.json"):

        from multiprocess import Process, Manager
        def f(d, min_cell_index,cropped_gt, cropped_candidate):
            d[str(min_cell_index)] = MapCell(min_cell_index,cropped_gt, cropped_candidate, self.options)
            print(d[str(min_cell_index)].metrics)

        manager = Manager()
        d = manager.dict()

        job = []
        # Add tqdm to show progress
        for min_cell_index, max_cell_index in self.iterate_cells():
        # for min_cell_index, max_cell_index in self.iterate_cells():
            
            cropped_gt, _ = get_cropped_point_cloud(self.pointcloud_GT, self.min_bound, self.cell_size, min_cell_index, max_cell_index)
            cropped_candidate, _ = get_cropped_point_cloud(self.pointcloud_Cnd, self.min_bound, self.cell_size, min_cell_index, max_cell_index)
            if cropped_gt.is_empty() and cropped_candidate.is_empty():
                pass
            else:
                job.append(Process(target=f, args=(d, min_cell_index,cropped_gt, cropped_candidate)))

        _ = [p.start() for p in job]
        _ = [p.join() for p in job]

        self.metriccells= copy.deepcopy(d)

        self.save_metric(filename)
            

   