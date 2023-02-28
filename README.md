# PQM: A Point Quality Evaluation Metric for Dense Maps

This repository implements the algorithms described in our paper [PQM: A Point Quality Evaluation Metric for Dense Maps]().

This is a point cloud quality evaluation metric that evaluates point clouds based on completeness, artifacts, accuracy and resolution.

Here we provide a fast implementation of PQM using [Open3d](http://www.open3d.org/), [PDAL](https://pdal.io/en/latest/download.html) and [PyTorch](https://pytorch.org/)

### Requirements

To use the CPU version of PQM, you need to have `python 3.8` and `pip3` installed. Next just run :

```
pip3 install -r requirements.txt
```
Install [PDAL](https://pdal.io/en/latest/download.html) for computing reference metrics 

For accelerating with CUDA (Instructions coming soon)

### Datasets

Sample data is located in `sample/`, additional datasets can be found [here](https://buffalo.box.com/s/h3a7lb7tlb82t63e7b15vgtz7p7tdemc)

### Preparing your `config` 

To compute PQM you need to create a `.json` configuration file. Similar to this :

```
{
    "gt_file": <path to source point cloud>,
    "cnd_file": [
        <path to candidate point cloud>
        
    ],
    "save_path": "../results/sample/",
    "cell_size": [
        5
    ],
    "weights": [ 
        [0.25,0.25, 0.25,0.25]
    ],
    "eps": [
        0.1
    ]
}
```

You may add as many `candidates`, `cell_sizes`, `weights` and `eps` as you want seperated by `,`

For example

```
{
    "gt_file": <path to source point cloud>,
    "cnd_file": [
        <path to candidate point cloud 1>,
        <path to candidate point cloud 2>,
        <path to candidate point cloud 3>
        
    ],
    "save_path": "../results/sample/",
    "cell_size": [
        5,10,15
    ],
    "weights": [ 
        [0.25,0.25, 0.25,0.25],
        [0.1,0.1, 0.4,0.4]
    ],
    "eps": [
        0.1,0.01
    ]
}
```


**Note :** PQM will be computed for each combination of set parameters

### Running `PQM`

This will take some time ...

```
python3 scripts/eval_config.py --config evaluation/sample.json --cpu
```

Replace the `sample.json` with your own `.json`

Flags for compute :
- `--cpu` - Single core computation (You may want to use this if you have large point clouds and less memory)
- `--cpu_multi` - Multi core computation (Parallelizes a single cell at a time, fully parallel code coming soon!)
- `--gpu_batch` - Pytorch implementation



### Running reference metrics (Chamfer and Hausdorff)

This will take some time ...

```
python3 scripts/ref_eval_config.py --config evaluation/sample.json --num_processes 2
```
Replace the `sample.json` with your own `.json` and set `--num_processes` as desired


### Inspecting Results

Results are stored at the location set in `.json` config file

Sample output for PQM : (Check 20 last lines)
```
 "gt_file": "../sample/subsample_noise_01_20_220_441_gt.pcd",
    "cnd_file": "../sample/subsample_noise_01_20_220_441_cnd.pcd",
    "average": {
        "completeness": 0.9968989102852018,
        "artifacts": 0.9999656018986713,
        "resolution": 0.9685709501090928,
        "accuracy": 0.848857745034264,
        "quality": 0.9535733018318077
    },
    "variance": {
        "completeness": 1.8494461128546425e-05,
        "artifacts": 3.6716016004050924e-09,
        "resolution": 0.0011251474913476901,
        "accuracy": 5.266424767623777e-06,
        "quality": 7.718279056574745e-05
    },
    "density_gt": 76.92592683982976,
    "density_cnd": 76.43603492769773,
    "chamfer": -1,
    "normalized_chamfer": -1,
    "hausdorff": -1
```

**Note :** CD,HD,NCD computation is commented for speed, use [this](#running-reference-metrics-chamfer-and-hausdorff) instead

### Visualizing Heatmaps

Coming soon ...

### Ablation usage

Coming soon ...



## Citation

If you use this library for any academic work, please cite the original [paper][].

```bibtex
Coming soon ...
```

