# slam_mapcomp

# testing preprocess on sample

Following command will create subsample with voxel size of 5 meter and display cropped version based on min_cell and max_cell

```python scripts/preprocess.py --gt sample/subsample_noise_01_20_220_441_gt.pcd --cnd sample/subsample_noise_01_20_220_441_cnd.pcd  --sub_sample --size 5 --filename subsample_noise_01_5_220_443 --min_cell 2,2,0 --max_cell 4,4,3```

To save the cropped version simply add "--save" command line argument to save on to filename defined by "--filename" argument