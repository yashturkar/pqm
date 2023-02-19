
from DamageManager import DamageManager , load_point_cloud
import argparse

from MapMetricManager import MapMetricManager
from system_constants import *

from QualityMetric import calculate_average_distance

from util import campute_image_pcd

def main():
    parser = argparse.ArgumentParser(description='Damage point cloud')
    parser.add_argument('pcd', type=str, help='Path to point cloud')
    parser.add_argument('damage', type=str, help='Type of damage to apply')
    parser.add_argument('--sigma', type=float, default="0.1", help='Standard deviation of gaussian noise in x,y,z equally direction')
    parser.add_argument('--percentage', type=float, default=0.1, help='Percentage of points to remove')
    parser.add_argument('--corner', default='top_left', type=str, help='Corner to remove points from')
    parser.add_argument('--voxel_size', type=float, default=0.1, help='Size of voxel to use for downsampling')
    parser.add_argument('--visualize', action='store_true', help='Visualize point cloud')
    parser.add_argument('--save', action='store_true', help='Save point cloud')
    parser.add_argument('--output', type=str, default="results/test.ply", help='Path to save point cloud')
    parser.add_argument('--size', type=float, default=0.1, help='cell size')
    args = parser.parse_args()

    # Load point cloud
    pcd = load_point_cloud(args.pcd)
    
    # Create damage manager
    damage_manager = DamageManager(pcd, args.size)

    damage_params=[]

    # Damage point cloud
    if args.damage == "gaussian":
        sigma = args.sigma
        print("sigma: ", sigma)
        damage_params = [sigma, sigma, sigma]
    elif args.damage == "random":
        damage_params = [args.percentage]
    elif args.damage == "remove":
        damage_params = [args.percentage]
    elif args.damage == "voxel":
        damage_params = [args.voxel_size]
    elif args.damage == "corners":
        damage_params = [args.corner]

    damage_pcd = damage_manager.damage_point_cloud(args.damage, damage_params)
    if args.visualize:
        damage_manager.savePointcloud(damage_pcd, args.output)
        metric_options = {WEIGHT_COMPLETENESS_STR:0.1, WEIGHT_ARTIFACTS_STR:0.1, WEIGHT_ACCURACY_STR:0.4,WEIGHT_RESOLUTION_STR:0.4, EPSILON_STR: 0.1}

        metric_manager = MapMetricManager(args.pcd, args.output, args.size, metric_options)

        avg_dist = calculate_average_distance(metric_manager.pointcloud_GT)
        print("Average distance: ", avg_dist/2)
        metric_manager.options[EPSILON_STR] = avg_dist/2

        metric_manager.compute_metric("results/bunny_damage_test.json")

        gt_vs_cnd_pcd = metric_manager.visualize(show=False, show_grid=True)
        heatmap_pcd = metric_manager.visualize_heatmap(QUALITY_STR, show=False, show_grid=True)
        
        campute_image_pcd(gt_vs_cnd_pcd, "results/bunny_damage_test_gt_vs_cnd.png")
        campute_image_pcd(heatmap_pcd, "results/bunny_damage_test_heatmap.png")


    if args.save:
        damage_manager.savePointcloud(damage_pcd, args.output)

if __name__ == "__main__":
    main()