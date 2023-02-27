# Script to extract data from multiple json files and save it in a csv file
import json
import os
import argparse
import pandas as pd


# Two types of json files:
# 1. json file for reference named ref, example name - village_fast_ref_metrics.json
# 2. json file for PQM, example name - village_fast_size_10_wc_0.25_wt_0.25_wa_0.25_wr_0.25_eps_0.1.json

# Function to extract name from json file, example "village_fast"
def extract_name(file_name):
    name = file_name.split("_")
    name = name[0] + "_" + name[1]
    return name

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True, help="Path to json files")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save csv file")
    args = parser.parse_args()
    json_path = args.json_path
    save_path = args.save_path

    # Get all json files in the path
    json_files = [pos_json for pos_json in os.listdir(json_path) if pos_json.endswith('.json')]

    print ("foo")

    # Create a dataframe to store data
    df = pd.DataFrame(columns=['file_name', 'chamfer_distance', 'hausdorff_distance', 'total_gt_points', 'total_cnd_points', 'normalized_chamfer_distance'])

    # Iterate over all json files
    for file in json_files:
        print (file)
        # Read json file
        with open(os.path.join(json_path, file)) as f:
            data = json.load(f)
        file_name = extract_name(file)
        chamfer_distance = data['chamfer']
        hausdorff_distance = data['hausdorff']
        total_gt_points = data['total_gt']
        total_cnd_points = data['total_cnd']
        normalized_chamfer_distance = data['normalized_chamfer']
        size = data['cell_size']
        options = data['options']
        completeness = data['average']['completeness']
        artifact_score = data['average']['artifacts']
        resolution = data['average']['resolution']
        accuracy = data['average']['accuracy']
        quality = data['average']['quality']
        variance = data['variance']
        # Add data to dataframe using pandas concat
        df = pd.concat([df, pd.DataFrame({'file_name': [file_name], 'cell_size' : [size] , 'chamfer_distance': [chamfer_distance], 'hausdorff_distance': [hausdorff_distance], 'total_gt_points': [total_gt_points], 'total_cnd_points': [total_cnd_points], 'normalized_chamfer_distance': [normalized_chamfer_distance], 'options': [options], 'completeness': [completeness] , 'artifact_score': [artifact_score] , 'resolution':[resolution] , 'accuracy':[accuracy], 'quality':[quality] , 'variance':[variance] })], ignore_index=True)
    # Save dataframe as csv file
    df.to_csv(save_path, index=False)

if __name__ == "__main__":
    main()