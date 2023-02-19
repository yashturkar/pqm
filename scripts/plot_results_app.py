import matplotlib.pyplot as plt


import open3d as o3d
import json

import numpy as np
import os

import argparse
from system_constants import *


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
                gt_file, damage_type, damage_param = file_name.split("_")
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


                print("file_name: ", file_name)
                print("attributes: ", attributes)

    # Filter data base on attributes
    # attributes = [CELL_SIZE_STR, WEIGHT_ACCURACY_STR, WEIGHT_COMPLETENESS_STR, WEIGHT_ARTIFACTS_STR, WEIGHT_RESOLUTION_STR, EPSILON_STR, CONFIG_GT_FILE_STR, CONFIG_DAMAGE_STR, CONFIG_DAMAGE_PARAMS_STR]
    
    # filter damage type as gaussian
    new_data, new_attributes = filter_by_attribute(data, attributes, CONFIG_DAMAGE_STR, "gaussian")

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


    plt.figure(figsize=(15,15))
    plt.plot(damage_list, new_data[QUALITY_STR])
    plt.plot(damage_list, new_data[ACCURACY_STR])    
    plt.plot(damage_list, new_data[COMPELTENESS_STR])
    plt.plot(damage_list, new_data[ARTIFACTS_STR])
    plt.plot(damage_list, new_data[RESOLUTION_STR])
    plt.xlabel("Damage Parameter")
    plt.ylabel("Accuracy/Quality")
    
    plt.legend([QUALITY_STR, ACCURACY_STR, COMPELTENESS_STR, ARTIFACTS_STR, RESOLUTION_STR])
    plt.show()


if __name__ == '__main__':
    main()