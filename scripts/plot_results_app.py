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
    parser.add_argument("--path", type=str, help="results json path" , default="results/bunny/")

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
    
    for name, group in results_df[requred_columns].groupby(CONFIG_DAMAGE_STR):
        #print('--------------------')
        print(name, len(group))
        #print('--------------------')
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        X1 =[]
        Y1=[]
        Z1=[]
        X_1 =[]
        Y_1=[]
        Z_1=[]
        for name1, group1 in group.groupby(CELL_SIZE_STR):
            print(name1, len(group1))
            print('--------------------')
            

            #plt.plot(group1[[CONFIG_DAMAGE_PARAMS_STR]], group1[[ACCURACY_STR, COMPELTENESS_STR, ARTIFACTS_STR, RESOLUTION_STR, QUALITY_STR]], label=name1)
            group1.sort_values(by=CONFIG_DAMAGE_PARAMS_STR, ascending=False,inplace=True)
            group1.sort_values(by=QUALITY_STR, ascending=False,inplace=True)

            #print(group1)

            y = group1[[CONFIG_DAMAGE_PARAMS_STR]]
            z = group1[[ACCURACY_STR, COMPELTENESS_STR, ARTIFACTS_STR, RESOLUTION_STR, QUALITY_STR]]
            x = group1[[CELL_SIZE_STR]]

            #ax.plot(x,y,z)
            #X_1.append(x.to_numpy().flatten())
            #Y_1.append(y.to_numpy().flatten())
            

            print(x.shape, y.shape, z.shape)
            ax.plot(x.to_numpy().flatten(),y.to_numpy().flatten(),z.to_numpy()[:,0].flatten(), 'go-')#, label=ACCURACY_STR)
            ax.plot(x.to_numpy().flatten(),y.to_numpy().flatten(),z.to_numpy()[:,1].flatten(), 'rd-')#, label=COMPELTENESS_STR)
            ax.plot(x.to_numpy().flatten(),y.to_numpy().flatten(),z.to_numpy()[:,2].flatten(), 'bs-')#, label=ARTIFACTS_STR)
            ax.plot(x.to_numpy().flatten(),y.to_numpy().flatten(),z.to_numpy()[:,3].flatten(), 'y*-')#, label=RESOLUTION_STR)
            ax.plot(x.to_numpy().flatten(),y.to_numpy().flatten(),z.to_numpy()[:,4].flatten(), 'cx-')#, label=QUALITY_STR)
            plt.legend()#labels=['Accuracy', 'Completeness', 'Artifacts', 'Resolution', 'Quality'])
            plt.title(name)
            ax.set_ylabel(CONFIG_DAMAGE_PARAMS_STR, fontsize=20, rotation=150)
            ax.set_xlabel(CELL_SIZE_STR, fontsize=20)
            ax.set_zlabel(QUALITY_STR, fontsize=20, rotation=60)

            ax.set_label('Label via method')
            X,Y, Z = x.to_numpy().ravel(),y.to_numpy().ravel(),z.to_numpy().ravel()
            x,y = X,Y

            top = Z.ravel()
            bottom = np.zeros_like(top)
            width = depth = 1

            print(x, y, top)

            Z1.append(z)#Z1.append(top)
            X1.append(x)
            Y1.append(y)
            #ax.bar3d(x, y, bottom, width, depth, top, shade=True)
            #ax.set_title('Shaded')
            
            # ax.plot_surface(x,y,z)
            # ax.plot_surface(x,group1[[ACCURACY_STR]],z)
            # ax.plot_surface(x,group1[[COMPELTENESS_STR]],z)
            # ax.plot_surface(x,group1[[ARTIFACTS_STR]],z)
            # ax.plot_surface(x,group1[[RESOLUTION_STR]],z)
            # ax.plot_surface(x,group1[[QUALITY_STR]],z)

        plt.legend(labels=['Accuracy', 'Completeness', 'Artifacts', 'Resolution', 'Quality'])

        # ax.plot(x.to_numpy().flatten(),y.to_numpy().flatten(),z.to_numpy()[:,0].flatten(), marker='o', label=ACCURACY_STR)
        # ax.plot(x.to_numpy().flatten(),y.to_numpy().flatten(),z.to_numpy()[:,1].flatten(), marker='d', label=COMPELTENESS_STR)
        # ax.plot(x.to_numpy().flatten(),y.to_numpy().flatten(),z.to_numpy()[:,2].flatten(), marker='s', label=ARTIFACTS_STR)
        # ax.plot(x.to_numpy().flatten(),y.to_numpy().flatten(),z.to_numpy()[:,3].flatten(), marker='*', label=RESOLUTION_STR)
        # ax.plot(x.to_numpy().flatten(),y.to_numpy().flatten(),z.to_numpy()[:,4].flatten(), marker='x', label=QUALITY_STR)
        plots_path = os.path.join(args.path,"plots")
        if not os.path.exists(plots_path):
            os.makedirs(plots_path)
            
        plt.savefig(os.path.join(plots_path,"{}_all.pdf".format(damage_type)))

        plt.show()

        # width = depth = 0.001

        # for metric in [ACCURACY_STR, COMPELTENESS_STR, ARTIFACTS_STR, RESOLUTION_STR, QUALITY_STR]:
        #     Z11 = []
        #     for i in range(len(Z1)):
        #         Z11.append(Z1[i][[metric]])
        #     #Z11 = Z1   
        #     bottom = np.zeros_like(np.array(Z11).ravel())
        #     ax.bar3d(np.array(Y1).ravel(),np.array(X1).ravel(),bottom , width, depth, np.array(Z11).ravel(), shade=True)
        #     #plt.zlabel(CELL_SIZE_STR)    
        #     plt.title(name)
        #     # plt.xlabel(CONFIG_DAMAGE_PARAMS_STR)
        #     # plt.ylabel(CELL_SIZE_STR)
        #     # plt.zlabel(QUALITY_STR)
        #     #plt.legend()
        #     ax.set_xlabel(CONFIG_DAMAGE_PARAMS_STR, fontsize=20, rotation=150)
        #     ax.set_ylabel(CELL_SIZE_STR, fontsize=20)
        #     ax.set_zlabel(QUALITY_STR, fontsize=20, rotation=60)
        #     print(np.array(X1).ravel(), np.array(Y1).ravel(), np.array(Z11).ravel())
        #     plt.show()
        #     plt.clf()

        
        

    return


    

if __name__ == '__main__':
    main()