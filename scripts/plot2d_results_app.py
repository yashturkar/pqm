import matplotlib.pyplot as plt


import open3d as o3d
import json

import numpy as np
import os

import argparse
from system_constants import *


import pandas as pd

from mpl_toolkits.mplot3d import Axes3D



def plot_2d(ax, x,y, title, xlabel, ylabel, legend=None):

    overlapping = 0.5
    ax.plot(x.to_numpy().flatten(),y.to_numpy()[:,0].flatten(), 'go-', alpha=overlapping, lw=5)#, label=ACCURACY_STR)
    ax.plot(x.to_numpy().flatten(),y.to_numpy()[:,1].flatten(), 'rd-', alpha=overlapping, lw=4)#, label=COMPELTENESS_STR)
    ax.plot(x.to_numpy().flatten(),y.to_numpy()[:,2].flatten(), 'bs-', alpha=overlapping, lw=3)#, label=ARTIFACTS_STR)
    ax.plot(x.to_numpy().flatten(),y.to_numpy()[:,3].flatten(), 'y*-', alpha=overlapping, lw=2)#, label=RESOLUTION_STR)
    ax.plot(x.to_numpy().flatten(),y.to_numpy()[:,4].flatten(), 'cx-', alpha=overlapping, lw=1)#, label=QUALITY_STR)
    ax.set_ylabel(ylabel, fontsize=28)
    ax.set_xlabel(xlabel, fontsize=28)
    ax.set_title(title, fontsize=28)

    if legend:
        ax.legend(legend, loc='upper left', fontsize=20)



def plot_2d_bar(ax, x,y, title, xlabel, ylabel, legend=None):
    title_map = {
        "voxel": "Downsample",
        "add": "Add",
        "remove": "Remove",
        "gaussian": "Gaussian",
        CONFIG_DAMAGE_PARAMS_STR: "Damage",
        QUALITY_STR: "Quality Metric",

    }

    width = 0.1
    if title=="voxel":
        width = 0.15
        ylabel = title_map[ylabel] + "(Sample Rate)"
    else:
        width = 0.005
        ylabel = title_map[ylabel] + "(%)"

    xlabel=    title_map[xlabel]

    title = title_map[title] + "(Cell Size: 0.05)"
    plt.bar(x.to_numpy().flatten()        ,y.to_numpy()[:,0].flatten(), width, color='g')#, label=ACCURACY_STR)
    plt.bar(x.to_numpy().flatten()+width*1,y.to_numpy()[:,1].flatten(), width, color='r')#, label=COMPELTENESS_STR)
    plt.bar(x.to_numpy().flatten()+width*2,y.to_numpy()[:,2].flatten(), width, color='b')#, label=ARTIFACTS_STR)
    plt.bar(x.to_numpy().flatten()+width*3,y.to_numpy()[:,3].flatten(), width, color='y')#, label=RESOLUTION_STR)
    plt.bar(x.to_numpy().flatten()+width*4,y.to_numpy()[:,4].flatten(), width, color='c',)#, label=QUALITY_STR)
    plt.ylabel(xlabel, fontsize=28)
    plt.xlabel(ylabel, fontsize=28)
    plt.title(title, fontsize=28)

    plt.xticks(x.to_numpy().flatten()+width*2, fontsize=25)
    plt.yticks(np.arange(0, 1.5, 0.1), fontsize=25)

    plt.ylim([0,1.5])
    if legend:
        plt.legend(legend, loc='upper left', fontsize=20)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="results json path" , default="results/bunny/")
    parser.add_argument("--cell_size", type=float, help="cell_size" , default=0.05)

    args = parser.parse_args()


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
                
                gt_file, damage_type, damage_param, _, _, _ = file_name.split("_")  #gt_basename, damagetype, damage_params, cell_size, weight, eps_
                attrib_dict[CONFIG_GT_FILE_STR] = gt_file
                attrib_dict[CONFIG_DAMAGE_STR] = damage_type
                attrib_dict[CONFIG_DAMAGE_PARAMS_STR] = float(damage_param)

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

    #results_df.set_index('Date', inplace=True)
    requred_columns = [ACCURACY_STR, COMPELTENESS_STR, ARTIFACTS_STR, RESOLUTION_STR, QUALITY_STR, CONFIG_DAMAGE_STR ,CONFIG_DAMAGE_PARAMS_STR, CELL_SIZE_STR]

    results_req_columns = results_df[requred_columns].groupby(CELL_SIZE_STR)
    
    #fig = plt.figure(figsize=(20, 10))
    
    #fig, ax_list = plt.subplots(1, 4, projection='3d')

    indx_ax=1
    for name, group in results_df[requred_columns].groupby(CONFIG_DAMAGE_STR):
        #print('--------------------')
        print(name, len(group))
        #print('--------------------')
        #ax = fig.add_subplot(1, 4, indx_ax, projection='3d')

        fig = plt.figure(figsize=(30, 20))
        ax = fig.add_subplot()


        for cell_size_entry, group1 in group.groupby(CELL_SIZE_STR):
            print(cell_size_entry, len(group1))
            print('--------------------')
            

            if cell_size_entry == args.cell_size:

                #plt.plot(group1[[CONFIG_DAMAGE_PARAMS_STR]], group1[[ACCURACY_STR, COMPELTENESS_STR, ARTIFACTS_STR, RESOLUTION_STR, QUALITY_STR]], label=name1)
                group1.sort_values(by=CONFIG_DAMAGE_PARAMS_STR, ascending=False,inplace=True)
                group1.sort_values(by=QUALITY_STR, ascending=False,inplace=True)

                #print(group1)

                x = group1[[CONFIG_DAMAGE_PARAMS_STR]]
                y = group1[[ACCURACY_STR, COMPELTENESS_STR, ARTIFACTS_STR, RESOLUTION_STR, QUALITY_STR]]
                

                print(x.shape, y.shape)


                plot_2d_bar(ax, x, y, name, QUALITY_STR, CONFIG_DAMAGE_PARAMS_STR, [ACCURACY_STR, COMPELTENESS_STR, ARTIFACTS_STR, RESOLUTION_STR, QUALITY_STR])

                plots_path = os.path.join(args.path,"plots")
                if not os.path.exists(plots_path):
                    os.makedirs(plots_path, exist_ok=True)
                print(name)
                plt.savefig(os.path.join(plots_path,"{}_2d_all.pdf".format(name)))
                plt.savefig(os.path.join(plots_path,"{}_2d_all.png".format(name)))
                
                ax.legend(labels=['Accuracy', 'Completeness', 'Artifacts', 'Resolution', 'Quality'], loc='upper right', fontsize=28)
                plt.show()

        indx_ax+=1


        
        

    return


    

if __name__ == '__main__':
    main()