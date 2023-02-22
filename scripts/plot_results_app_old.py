import matplotlib.pyplot as plt


import open3d as o3d
import json

import numpy as np
import os

import argparse
from system_constants import *


import pandas as pd

def filter_by_attribute(data, attributes, attribute_name, attribute_value):

    # filter damage type as gaussian
    new_data = {}
    new_data[COMPELTENESS_STR]=[]
    new_data[ACCURACY_STR]=[]
    new_data[ARTIFACTS_STR]=[]
    new_data[RESOLUTION_STR]=[]
    new_data[QUALITY_STR]=[]
    new_data[HOUSDORFF_STR]=    []
    new_data[NORMALIZED_CHAMFER_STR]=[]
    new_data[CHAMFER_STR]=[]
    new_attributes = []

    for attrib_row in attributes:
        if attrib_row[attribute_name] == attribute_value:
            new_data[COMPELTENESS_STR].append(data[COMPELTENESS_STR][attributes.index(attrib_row)])
            new_data[ACCURACY_STR].append(data[ACCURACY_STR][attributes.index(attrib_row)])
            new_data[ARTIFACTS_STR].append(data[ARTIFACTS_STR][attributes.index(attrib_row)])
            new_data[RESOLUTION_STR].append(data[RESOLUTION_STR][attributes.index(attrib_row)])
            new_data[QUALITY_STR].append(data[QUALITY_STR][attributes.index(attrib_row)])
            new_data[HOUSDORFF_STR].append(data[HOUSDORFF_STR][attributes.index(attrib_row)])
            new_data[NORMALIZED_CHAMFER_STR].append(data[NORMALIZED_CHAMFER_STR][attributes.index(attrib_row)])
            new_data[CHAMFER_STR].append(data[CHAMFER_STR][attributes.index(attrib_row)])
            new_attributes.append(attrib_row)

    return new_data, new_attributes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="results json path" , default="results/bunny/")

    args = parser.parse_args()

    data = {}
    data[COMPELTENESS_STR]=[]
    data[ACCURACY_STR]=[]
    data[ARTIFACTS_STR]=[]
    data[RESOLUTION_STR]=[]
    data[QUALITY_STR]=[]
    data[HOUSDORFF_STR]=    []
    data[NORMALIZED_CHAMFER_STR]=[]
    data[CHAMFER_STR]=[]

    attributes = []

    results = {}

    results[COMPELTENESS_STR]=[]
    results[ACCURACY_STR]=[]
    results[ARTIFACTS_STR]=[]
    results[RESOLUTION_STR]=[]
    results[QUALITY_STR]=[]
    results[HOUSDORFF_STR]=    []
    results[NORMALIZED_CHAMFER_STR]=[]
    results[CHAMFER_STR]=[]

    results[CELL_SIZE_STR] =[]
    results[WEIGHT_ACCURACY_STR] = []
    results[WEIGHT_COMPLETENESS_STR] = []
    results[WEIGHT_ARTIFACTS_STR] = []
    results[WEIGHT_RESOLUTION_STR] = []
    results[EPSILON_STR] = []
    results[CONFIG_GT_FILE_STR] = []
    results[CONFIG_DAMAGE_STR] = []
    results[CONFIG_DAMAGE_PARAMS_STR] = []




    for path in os.listdir(args.path):
        if path.endswith(".json"):
            with open(os.path.join(args.path, path)) as f:
                data_load = json.load(f)
                attrib_dict={}
                attrib_dict[CELL_SIZE_STR] = data_load[CELL_SIZE_STR]
               
                attrib_dict[WEIGHT_ACCURACY_STR] = data_load[CONFIG_OPTIONS_STR][WEIGHT_ACCURACY_STR]
                attrib_dict[WEIGHT_COMPLETENESS_STR] = data_load[CONFIG_OPTIONS_STR][WEIGHT_COMPLETENESS_STR]
                attrib_dict[WEIGHT_ARTIFACTS_STR] = data_load[CONFIG_OPTIONS_STR][WEIGHT_ARTIFACTS_STR]
                attrib_dict[WEIGHT_RESOLUTION_STR] = data_load[CONFIG_OPTIONS_STR][WEIGHT_RESOLUTION_STR]
                attrib_dict[EPSILON_STR] = data_load[CONFIG_OPTIONS_STR][EPSILON_STR]
                file_name = os.path.basename(path)[:-5]
                #gt_basename, damagetype, damage_params, cell_size, weight, eps_
                gt_file, damage_type, damage_param, _, _, _ = file_name.split("_")
                attrib_dict[CONFIG_GT_FILE_STR] = gt_file
                attrib_dict[CONFIG_DAMAGE_STR] = damage_type
                attrib_dict[CONFIG_DAMAGE_PARAMS_STR] = float(damage_param)
                attributes.append(attrib_dict)

                data[COMPELTENESS_STR].append(data_load[AVERAGE_STR][COMPELTENESS_STR])
                data[ACCURACY_STR].append(data_load[AVERAGE_STR][ACCURACY_STR])
                data[ARTIFACTS_STR].append(data_load[AVERAGE_STR][ARTIFACTS_STR])
                data[RESOLUTION_STR].append(data_load[AVERAGE_STR][RESOLUTION_STR])
                data[QUALITY_STR].append(data_load[AVERAGE_STR][QUALITY_STR])
                data[HOUSDORFF_STR].append(data_load[HOUSDORFF_STR])
                data[NORMALIZED_CHAMFER_STR].append(data_load[NORMALIZED_CHAMFER_STR])
                data[CHAMFER_STR].append(data_load[CHAMFER_STR])

                results[COMPELTENESS_STR].append(data_load[AVERAGE_STR][COMPELTENESS_STR])
                results[ACCURACY_STR].append(data_load[AVERAGE_STR][ACCURACY_STR])
                results[ARTIFACTS_STR].append(data_load[AVERAGE_STR][ARTIFACTS_STR])
                results[RESOLUTION_STR].append(data_load[AVERAGE_STR][RESOLUTION_STR])
                results[QUALITY_STR].append(data_load[AVERAGE_STR][QUALITY_STR])
                results[HOUSDORFF_STR].append(data_load[HOUSDORFF_STR])
                results[NORMALIZED_CHAMFER_STR].append(data_load[NORMALIZED_CHAMFER_STR])
                results[CHAMFER_STR].append(data_load[CHAMFER_STR])

                for attrib in attrib_dict:
                    results[attrib].append(attrib_dict[attrib])
                

                print("file_name: ", file_name)
                #print("attributes: ", attributes)


    results_df = pd.DataFrame.from_dict(results)

    print(results_df.head())

    return
    # Filter data base on attributes
    # attributes = [CELL_SIZE_STR, WEIGHT_ACCURACY_STR, WEIGHT_COMPLETENESS_STR, WEIGHT_ARTIFACTS_STR, WEIGHT_RESOLUTION_STR, EPSILON_STR, CONFIG_GT_FILE_STR, CONFIG_DAMAGE_STR, CONFIG_DAMAGE_PARAMS_STR]
    print("Filtering data based on attributes")
    
    damage_type_to_metric_map = {
        GAUSSIAN_STR: ACCURACY_STR,
        ADD_POINTS_STR: ARTIFACTS_STR,
        REMOVE_POINTS_STR: COMPELTENESS_STR,
        DOWNSAMPLE_STR: RESOLUTION_STR
    }

    damage_type_to_string_map = {
        GAUSSIAN_STR: "Gaussian (\mu=0, \sigma)",
        ADD_POINTS_STR: "Add Points (\%)",
        REMOVE_POINTS_STR: "Remove Points (\%)",
        DOWNSAMPLE_STR: "Downsample (voxel size)"
    }

    cell_sizes = []
    for attrib_row in attributes:
        if attrib_row[CELL_SIZE_STR] not in cell_sizes:
            cell_sizes.append(attrib_row[CELL_SIZE_STR])
    cell_sizes.sort()
    
    for cell_size in cell_sizes:
        for damage_type in DAMAGE_TYPES_LIST:

            # filter damage type as gaussian
            new_data, new_attributes = filter_by_attribute(data, attributes, CONFIG_DAMAGE_STR, damage_type)
            new_data, new_attributes = filter_by_attribute(new_data, new_attributes, CELL_SIZE_STR, cell_size)
            damage_list = []
            for attrib in new_attributes:
                damage_list.append(attrib[CONFIG_DAMAGE_PARAMS_STR])

            #sort data based on damage param
            new_data[COMPELTENESS_STR] = [x for _,x in sorted(zip(damage_list, new_data[COMPELTENESS_STR]))]
            new_data[ACCURACY_STR] = [x for _,x in sorted(zip(damage_list, new_data[ACCURACY_STR]))]
            new_data[ARTIFACTS_STR] = [x for _,x in sorted(zip(damage_list, new_data[ARTIFACTS_STR]))]
            new_data[RESOLUTION_STR] = [x for _,x in sorted(zip(damage_list, new_data[RESOLUTION_STR]))]
            new_data[QUALITY_STR] = [x for _,x in sorted(zip(damage_list, new_data[QUALITY_STR]))]

            damage_list.sort()
            plots_path = os.path.join(args.path,"plots")
            if not os.path.exists(plots_path):
                os.makedirs(plots_path)

            plt.figure(figsize=(15,15))
            plt.plot(damage_list, new_data[QUALITY_STR])
            plt.plot(damage_list, new_data[ACCURACY_STR])    
            plt.plot(damage_list, new_data[COMPELTENESS_STR])
            plt.plot(damage_list, new_data[ARTIFACTS_STR])
            plt.plot(damage_list, new_data[RESOLUTION_STR])
            plt.xlabel("Damage Parameter : {}".format(damage_type_to_string_map[damage_type]))
            plt.ylabel("Metric")
            plt.title("Damage Type : {}".format(damage_type))
            plt.legend([QUALITY_STR, ACCURACY_STR, COMPELTENESS_STR, ARTIFACTS_STR, RESOLUTION_STR])


            plt.savefig(os.path.join(plots_path,"{}_{}_All.png".format(cell_size, damage_type)))

            plt.show()

            plt.figure(figsize=(15,15))
            plt.plot(damage_list, new_data[QUALITY_STR])
            plt.plot(damage_list, new_data[damage_type_to_metric_map[damage_type]])    
            plt.xlabel("Damage Parameter : {}".format(damage_type_to_string_map[damage_type]))
            plt.ylabel(damage_type_to_metric_map[damage_type])
            plt.title("Damage Type : {}".format(damage_type))
            plt.legend([QUALITY_STR, damage_type_to_metric_map[damage_type]])
        
            plt.savefig(os.path.join(plots_path,"{}_{}_{}.png".format(cell_size, damage_type, damage_type_to_metric_map[damage_type])))

            plt.show()  



    

if __name__ == '__main__':
    main()