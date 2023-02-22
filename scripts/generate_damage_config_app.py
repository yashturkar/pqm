
import argparse
import json
import os

from MapMetricManager import MapMetricManager, parse_mapmetric_config
from DamageManager import load_point_cloud, DamageManager
from QualityMetric import calculate_average_distance
from util import campute_image_pcd

from system_constants import *

from util import Timer

    # parser.add_argument('pcd', type=str, help='Path to point cloud')
    # parser.add_argument('damage', type=str, help='Type of damage to apply')
    # parser.add_argument('--sigma', type=float, default="0.1", help='Standard deviation of gaussian noise in x,y,z equally direction')
    # parser.add_argument('--percentage', type=float, default=0.1, help='Percentage of points to remove')
    # parser.add_argument('--corner', default='top_left', type=str, help='Corner to remove points from')
    # parser.add_argument('--voxel_size', type=float, default=0.1, help='Size of voxel to use for downsampling')
    # parser.add_argument('--visualize', action='store_true', help='Visualize point cloud')
    # parser.add_argument('--save', action='store_true', help='Save point cloud')
    # parser.add_argument('--output', type=str, default="results/test.ply", help='Path to save point cloud')
    # parser.add_argument('--size', type=float, default=0.1, help='cell size')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config json file")

    # Add 4 options for compute mode, gpu, cpu, gpu-batch, cpu-multi
    compute_mode = parser.add_mutually_exclusive_group()
    compute_mode.add_argument("--gpu", action="store_true", help="compute mode gpu")
    compute_mode.add_argument("--cpu", action="store_true", help="compute mode cpu")
    compute_mode.add_argument("--gpu_batch", action="store_true", help="compute mode gpu-batch")
    compute_mode.add_argument("--cpu_multi", action="store_true", help="compute mode cpu-multi")

    parser.add_argument("--genviz", help="generate viz/ heatmap image", action="store_true")

    # Set compute flag 1 for gpu, 2 for cpu, 3 for gpu-batch, 4 for cpu-multi
    compute_flag = 0
    if parser.parse_args().gpu:
        compute_flag = 1
    elif parser.parse_args().cpu:
        compute_flag = 2
    elif parser.parse_args().gpu_batch:
        compute_flag = 3
    elif parser.parse_args().cpu_multi:
        compute_flag = 4

    args = parser.parse_args()
    config = {}
    with open(args.config, mode='r') as fp:
        config = json.load(fp)

        gt_path = config[CONFIG_GT_FILE_STR]

        save_path = config[CONFIG_SAVE_PATH_STR]

        cell_sizes = config[CELL_SIZE_STR]#[0]
        weights = config[CONFIG_WEIGHTS_STR]#[0]
        eps = config[CONFIG_EPS_STR]#[0]


        # Load point cloud
        pcd = load_point_cloud(gt_path)

        avg_dist = calculate_average_distance(pcd)

        # print("eps: ", eps)
        # if eps is None:
        #     avg_dist = calculate_average_distance(pcd)
        #     print("Average distance: ", avg_dist/2)
        #     eps = avg_dist/2


        # metric_options = {WEIGHT_COMPLETENESS_STR: weights[0], WEIGHT_ARTIFACTS_STR: weights[1], WEIGHT_ACCURACY_STR: weights[2],WEIGHT_RESOLUTION_STR: weights[3], EPSILON_STR: eps}



        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(gt_path):
            print("gt path not exist")
            return
        gt_basename = os.path.basename(gt_path).split(".")[0]# + "_damaged.ply"
  
        damages = config[CONFIG_DAMAGE_STR]

        with Timer('Damage loop'):

            for cell_size in cell_sizes:
                # Create damage manager
                damage_manager = DamageManager(pcd, cell_size)
                for weight in weights:
                    for eps_ in eps:
                        if eps_ is None:
                            print("Average distance: ", avg_dist/2)
                            eps_ = avg_dist/2
                        metric_options = {WEIGHT_COMPLETENESS_STR: weight[0], WEIGHT_ARTIFACTS_STR: weight[1], WEIGHT_ACCURACY_STR: weight[2],WEIGHT_RESOLUTION_STR: weight[3], EPSILON_STR: eps_}
                        for damagetype in damages:
                            print("damagetype: ", damagetype)
                            for damage_params in damages[damagetype]:   
                                print("damagetype: ", damagetype, "param: ", damage_params)
                                with Timer(damagetype + '=' + str(damage_params) + ' cell_size=' + str(cell_size) + ' weight=' + str(weight) + ' eps=' + str(eps_)):
                                    curr_case_name = "{}_{}_{}_{}_{}_{}".format(gt_basename, damagetype, damage_params, cell_size, weight, eps_)
                                    results_found = False
                                    if os.path.exists(os.path.join(save_path, "{}.json".format(curr_case_name))):
                                        print("case already exist")
                                        results_found = True
                                        if not args.genviz:
                                            continue
                                    if not results_found:
                                        damage_pcd = damage_manager.damage_point_cloud(damagetype, damage_params)
                                                            
                                        temp_output = os.path.join(save_path, "{}.ply".format(curr_case_name))

                                        # Save point cloud temporarily for MapMetricManager
                                        damage_manager.savePointcloud(damage_pcd, temp_output)
                                        
                                        metric_manager = MapMetricManager(gt_path, temp_output, cell_size, metric_options, compute_flag=compute_flag)

                                        if np.any(damage_manager.cell_dim != metric_manager.cell_dim):
                                            print("Cell dimension mismatch : ", damage_manager.cell_dim, metric_manager.cell_dim)
                                            continue

                                        metric_manager.compute_metric(os.path.join(save_path, "{}.json".format(curr_case_name)))
                                    else:
                                        metric_manager = parse_mapmetric_config(os.path.join(save_path, "{}.json".format(curr_case_name)))

                                    if args.genviz:
                                        print("generate viz")
                                        gt_vs_cnd_pcd = metric_manager.visualize(show=False, show_grid=True)
                                        heatmap_pcd = metric_manager.visualize_heatmap(QUALITY_STR, show=False, show_grid=True)
                                        
                                        campute_image_pcd(gt_vs_cnd_pcd, os.path.join(save_path, "{}_gt_vs_cnd.png".format(curr_case_name)))
                                        campute_image_pcd(heatmap_pcd, os.path.join(save_path, "{}_heatmap.png".format(curr_case_name)))

                



if __name__ == "__main__":
    main()

