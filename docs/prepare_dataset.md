# Prepare S3DIS and ScanNetV2 Data
**1. Download [S3DIS](https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1) and [ScanNetV2](http://www.scan-net.org/) data.**

Uncompress them to `dataset/S3DIS` and `dataset/ScanNetV2`.


**2. Prepared training data**
```
python utils/data_prepare_s3dis.py       # prepare for S3DIS
python utils/data_prepare_scannetv2.py   # prepare for ScanNetV2
```

The data structure of the project is organized as:
```
dataset
└──S3DIS                                     #  S3DIS dataset
│   ├── input_0.040
│   │   ├── *.ply
│   │   ├── *_proj.pkl
│   │   └── *_KDTree.pkl
│   ├── original_ply
│   │   └── *.ply
│   │
│   └── Stanford3dDataset_v1.2_Aligned_Version
└──ScanNetV2                                 #  ScanNetV2 dataset
│   ├── input_0.040
│   │   ├── *.ply
│   │   ├── *_proj.pkl
│   │   └── *_KDTree.pkl
│   ├── original_ply
│   │   └── *.ply
│   │
│   ├── scans
│   ├── scans_test
│   ├── tasks
│   ├── scannet_v2_test.txt
│   ├── scannet_v2_val.txt
│   └── scannetv2-labels.combined.tsv
```
