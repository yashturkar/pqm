# Script to input 2 json files and merge them into one json file
import json
import os
import argparse


def merge_json(json1, json2):
    # Open json files
    with open(json1) as f1:
        data1 = json.load(f1)
    with open(json2) as f2:
        data2 = json.load(f2)
    # Merge json files
    data1.update(data2)
    return data1

def main():
    parser = argparse.ArgumentParser()
    # Go over all files in the folder and merge them based on the file name
    # File name should be 1st two words of the file name split by "_"
    parser.add_argument("--jsons", type=str, required=True, help="Path to json files directory")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save merged json files")
    args = parser.parse_args()
    # Get all json files in the directory
    json_files = [pos_json for pos_json in os.listdir(args.jsons) if pos_json.endswith('.json')]
    # Sort json files
    json_files.sort()

    # Make save path directory if it doesn't exist
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    for j1 in json_files:
        if j1.split("_")[2] == "ref":
            for j2 in json_files:
                if (j1.split("_")[0] + j1.split("_")[1] == j2.split("_")[0] + j2.split("_")[1]) and (j1 != j2):
                    # print (j1.split("_")[0] + j1.split("_")[1] + "   " + j2.split("_")[0] + j2.split("_")[1])
                    print ("Merging  ->  " +  j1 + "    with   " + j2)
                    # Remove chamfer, hausdorff and normalized chamfer from j2 file 
                    # because they are not needed for the reference file
                    with open(os.path.join(args.jsons, j2)) as f:
                        data = json.load(f)
                    data.pop("chamfer", None)
                    data.pop("hausdorff", None)
                    data.pop("normalized_chamfer", None)
                    with open(os.path.join(args.jsons, j2), mode='w') as fp:
                        json.dump(data, fp, indent=4)
                    name = j1.split("_")[0] + "_" + j1.split("_")[1]
                    # Merge json files
                    merged_json = merge_json(os.path.join(args.jsons, j1), os.path.join(args.jsons, j2))
                    # Save merged json file
                    with open(os.path.join(args.save_path, (name+".json")), mode='w') as fp:
                        json.dump(merged_json, fp, indent=4)
                else: 
                    continue

if __name__ == "__main__":
    main()