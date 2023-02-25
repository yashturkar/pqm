import matplotlib.pyplot as plt


import open3d as o3d
import json

import numpy as np
import os

import argparse
from system_constants import *


import pandas as pd

from mpl_toolkits.mplot3d import Axes3D


def main_():
    # set up the figure and axes
    fig = plt.figure(figsize=(8, 3))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    # fake data
    _x = np.arange(4)
    _y = np.arange(5)
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()

    top = x + y
    bottom = np.zeros_like(top)
    width = depth = 1

    ax1.bar3d(x, y, bottom, width, depth, top, shade=True)
    ax1.set_title('Shaded')

    print(x, y, bottom, width, depth, top)

    ax2.bar3d(x, y, bottom, width, depth, top, shade=False)
    ax2.set_title('Not Shaded')

    plt.show()

    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="results json path" , default="results/bunny_v6/")
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

    PLY_FILE_STR = "ply_file"
    results[PLY_FILE_STR] = []



    for path in os.listdir(args.path):
        if path.endswith(".json"):
            with open(os.path.join(args.path, path)) as f:

                curr_file_name = os.path.join(args.path, path)[:-(len(".json"))]+".ply"
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

                results[PLY_FILE_STR].append(curr_file_name)

                for attrib in attrib_dict:
                    results[attrib].append(attrib_dict[attrib])
                

                #print("file_name: ", file_name)
                #print("attributes: ", attributes)


    results_df = pd.DataFrame.from_dict(results)

    requred_columns = [CONFIG_DAMAGE_STR, CONFIG_DAMAGE_PARAMS_STR, CHAMFER_STR, HOUSDORFF_STR, QUALITY_STR, RESOLUTION_STR, ACCURACY_STR, COMPELTENESS_STR, ARTIFACTS_STR, CELL_SIZE_STR, PLY_FILE_STR]
    results_df = results_df[requred_columns]
    results_df_new = results_df[results_df[CELL_SIZE_STR] == args.cell_size]

    results_df_new  = results_df_new[results_df_new[CONFIG_DAMAGE_PARAMS_STR].isin([0.25,0.5,0.75,5,10,15])]

    results_df_new.sort_values(by=[CONFIG_DAMAGE_STR, CONFIG_DAMAGE_PARAMS_STR], inplace=True)

    #print(results_df_new[[PLY_FILE_STR]])
    count_cand = []
    count_gt = []
    gt_file_name = 'sample/bunny/bunny.ply'
    for i in range(len(results_df_new)):
        pcd = o3d.io.read_point_cloud(results_df_new.iloc[i][PLY_FILE_STR])
        pcd_gt = o3d.io.read_point_cloud(gt_file_name)
        count_cand.append(len(pcd.points))
        count_gt.append(len(pcd_gt.points))
        print(results_df_new.iloc[i][PLY_FILE_STR])
       
    CANDIDATE_COUNT_STR = "Candidate_Count"
    GT_COUNT_STR = "GT_Count"
    results_df_new[CANDIDATE_COUNT_STR] = count_cand
    results_df_new[GT_COUNT_STR] = count_gt
    print("count_cand: ", count_cand)
    print("count_gt: ", count_gt)
    results_df_new = results_df_new[[CONFIG_DAMAGE_STR, CONFIG_DAMAGE_PARAMS_STR, CANDIDATE_COUNT_STR, GT_COUNT_STR, CHAMFER_STR, HOUSDORFF_STR, QUALITY_STR, RESOLUTION_STR, ACCURACY_STR, COMPELTENESS_STR, ARTIFACTS_STR]]
    results_df_new.to_latex('results/bunny_v6/plots/latex_table.txt', index=False, header=True)
    print(results_df_new)
    return 
    #results_df.set_index('Date', inplace=True)
    requred_columns = [ACCURACY_STR, COMPELTENESS_STR, ARTIFACTS_STR, RESOLUTION_STR, QUALITY_STR, CONFIG_DAMAGE_STR ,CONFIG_DAMAGE_PARAMS_STR, CELL_SIZE_STR]


    filtered_df = [] #pd.DataFrame()
    indx_ax=1
    for name, group in results_df[requred_columns].groupby(CONFIG_DAMAGE_STR):
        #print('--------------------')
        #print(name, len(group))



        for name1, group1 in group.groupby(CELL_SIZE_STR):
            #print(name1, len(group1))
            #print('--------------------')
            
            if name1 == args.cell_size:
                #print(name1, len(group1))
                #print('--------------------')
                #print(group1)

                for name2, group2 in group1.groupby(CONFIG_DAMAGE_PARAMS_STR):
                    #print(name, name2, len(group2))
                    #print('--------------------')
                    #print(group2)
                    #print('--------------------')
                     
                    if name == "voxel" and name2 == [5, 10, 15]:
                        #print(group2)
                        filtered_df.append(group2)
                        #break
                    if name != "voxel" and name2 in [0.25, 0.5, 0.75]:
                        #print(group2)
                        filtered_df.append(group2)
                        
                        #break
                    


            
        #print(name)


        indx_ax+=1
    filtered_df = pd.concat(filtered_df)
    print(filtered_df)

    return


    

if __name__ == '__main__':
    main()



#  \textbf{Degrade Type} &  \textbf{Parameter} & Candidate Pts & GT pts & $D_{chamfer}$ & $D_{hausdorff}$ &  & $Q_{res}$ & $Q_{accr}$ & $Q_{comp}$ & $Q_{art}$ & $Q_{PQM}$ \\ 
# \hline

#      add &           0.25 &           213586 &    100106 & 10.519655 &   0.034282 &    0.858784 &  0.999903 &      1.000000 &   0.867539 & 0.931557 \\
#      add &           0.50 &           256309 &    100106 & 21.038810 &   0.036003 &    0.870882 &  0.999823 &      1.000000 &   0.769228 & 0.909983 \\
#      add &           0.75 &           299028 &    100106 & 31.570433 &   0.039840 &    0.885738 &  0.999792 &      1.000000 &   0.692642 & 0.894543 \\ \hline
# gaussian &           0.25 &           170877 &    100106 &  0.000451 &   0.000261 &    1.000000 &  0.904481 &      0.999937 &   0.999410 & 0.975957 \\
# gaussian &           0.50 &           170877 &    100106 &  0.000938 &   0.000267 &    1.000000 &  0.803757 &      0.999937 &   0.998539 & 0.950558 \\
# gaussian &           0.75 &           170877 &    100106 &  0.001458 &   0.000252 &    0.999882 &  0.705511 &      0.999809 &   0.994937 & 0.925035 \\ \hline
#   remove &           0.25 &           128168 &    100106 &  0.310759 &   0.014659 &    0.976786 &  1.000000 &      0.968762 &   1.000000 & 0.986387 \\
#   remove &           0.50 &            85445 &    100106 &  1.610939 &   0.019903 &    0.864487 &  1.000000 &      0.839478 &   1.000000 & 0.925991 \\
#   remove &           0.75 &            42726 &    100106 &  6.744597 &   0.024817 &    0.627799 &  1.000000 &      0.471273 &   1.000000 & 0.774768 \\ \hline
#    voxel &           5.00 &            34187 &    100106 &  0.045465 &   0.002689 &    0.466551 &  1.000000 &      0.439909 &   1.000000 & 0.726615 \\
#    voxel &          10.00 &            17099 &    100106 &  0.103087 &   0.003899 &    0.255276 &  1.000000 &      0.226599 &   1.000000 & 0.620469 \\
#    voxel &          15.00 &            11406 &    100106 &  0.162375 &   0.005424 &    0.148068 &  1.000000 &      0.149061 &   1.000000 & 0.574282 \\