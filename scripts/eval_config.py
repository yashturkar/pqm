
import argparse
import json
import os

from preprocess import MapMetricManager

# config["gt_path"] = "gt.pcd"
# config["cnd_paths"] = ["fast.pcd", "lego.pcd"]
# config["save_path"] = "results/"
# config["e"] = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
# config["MPD"] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# config["size"] = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]


# Wi -> Incompteness
# Wt -> Artifacts
# Wr -> Resolution
# Wa -> Accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config json file")

    args = parser.parse_args()
    config = {}
    with open(args.config, mode='r') as fp:
        config = json.load(fp)

        gt_path = config["gt_path"]
        cnd_paths = config["cnd_paths"]
        save_path = config["save_path"]
        e = config["e"]
        MPD = config["MPD"]
        size = config["size"]
        weights = config["w1"]

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(gt_path):
            print("gt path not exist")
            return
        
        for cnd_path in cnd_paths:
            if not os.path.exists(cnd_path):
                print("cnd path not exist")
                continue
            mapManager = None
            for e_ in e:
                for MPD_ in MPD:
                    for size_ in size:
                        for w1_ in weights:
                            metric_options = {"e": e_ , "MPD": MPD_, "w1": w1_}
                            if mapManager is None:
                                mapManager = MapMetricManager(gt_path, cnd_path, size_, metric_options=metric_options)
                            cmd_file_name = cnd_path.split("/")[-1].split(".")[0]
                            mapManager.reset(size_, metric_options=metric_options)
                            mapManager.compute_metric(os.path.join(save_path, "{}_e_{}_MPD_{}_size_{}_w1_{}.json".format(cmd_file_name, e_, MPD_, size_, w1_)))
        


        #mapManager = MapMetricManager(args.gt, args.cnd, args.size, metric_options=metric_options)



if __name__ == "__main__":
    main()

