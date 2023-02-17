import numpy as np

from preprocess import parse_mapmetric_config

import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config json file")

    args = parser.parse_args()

    mapmetric = parse_mapmetric_config(args.config)
    
    mapmetric.visualize_heatmap("quality")

if __name__ == "__main__":
    main()