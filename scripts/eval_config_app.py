
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

        size = config["size"]
        weights = config["weights"]

        eps = config["eps"]

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
            for size_ in size:
                for w1_ in weights:
                    for eps_ in eps:
                        metric_options = {"wc":w1_[0], "wt":w1_[1], "wa":w1_[2],"wr":w1_[3], "e": eps_}
                        
                        if mapManager is None:
                            mapManager = MapMetricManager(gt_path, cnd_path, size_, metric_options=metric_options)
                        cmd_file_name = cnd_path.split("/")[-1].split(".")[0]
                        mapManager.reset(size_, metric_options=metric_options)
                        mapManager.compute_metric(os.path.join(save_path, "{}_size_{}_wc_{}_wt_{}_wa_{}_wr_{}_eps_{}.json".format(cmd_file_name, size_, w1_[0],w1_[1],w1_[2],w1_[3]), eps_))
            


        #mapManager = MapMetricManager(args.gt, args.cnd, args.size, metric_options=metric_options)



if __name__ == "__main__":
    main()

