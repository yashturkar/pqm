import numpy as np

from MapMetricManager import parse_mapmetric_config

import argparse

from system_constants import *


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config json file")

    args = parser.parse_args()

    mapmetric = parse_mapmetric_config(args.config)
    
    mapmetric.visualize_heatmap(QUALITY_STR)

if __name__ == "__main__":
    main()