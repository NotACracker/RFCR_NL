# RFCR-NL: Positive-Negative Receptive Field Reasoning for Omni-supervised 3D Segmentation (TPAMI 2023)
This is the official implementation of RFCR-NL (TPAMI 2023), which is the extension work of RFCR (CVPR 2021) and utilized negative learning for both fully supervised and weakly supervised point cloud segmentation. For technical details, please refer to:

**Positive-Negative Receptive Field Reasoning for Omni-supervised 3D Segmentation (TPAMI 2023)**

Xin Tan*, Qihang Ma*, Jingyu Gong, Jiachen Xu, Zhizhong Zhang, Haichuan Song, Yanyun Qu, Yuan Xie, and Lizhuang Ma. (* co-first authors)

**[[Paper](https://ieeexplore.ieee.org/document/10264222)]**

**Omni-supervised Point Cloud Segmentation via Gradual Receptive Field Component Reasoning (CVPR 2021)**

Jingyu Gong, Jiachen Xu, Xin Tan, Haichuan Song, Yanyun Qu, Yuan Xie, Lizhuang Ma.

**[[Paper](https://arxiv.org/pdf/2105.10203.pdf)] [[Code](https://github.com/azuki-miho/RFCR)]**

![RFCR Framework](./figs/rfcrnl_framework.jpg)

## News
- `[2023/12/11]` Initial code release.
- :star2:`[2023/9/23]` RFCR-NL was accepted to TPAMI 2023.
  
### Getting Started

- [Install](docs/install.md)
- [Prepare dataset](docs/prepare_dataset.md)
- [Train&Validate](docs/train_val.md)

## Acknowledgement

Many thanks to these excellent open source projects:
- [RandLA-Net](https://github.com/QingyongHu/RandLA-Net/), [KPConv](https://github.com/HuguesTHOMAS/KPConv), [NLNL](https://github.com/ydkim1293/NLNL-Negative-Learning-for-Noisy-Labels).


## BibTex
If this work is helpful for your research, please consider citing:
```
@ARTICLE{tan2023positive,
  author={Tan, Xin and Ma, Qihang and Gong, Jingyu and Xu, Jiachen and Zhang, Zhizhong and Song, Haichuan and Qu, Yanyun and Xie, Yuan and Ma, Lizhuang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Positive-Negative Receptive Field Reasoning for Omni-Supervised 3D Segmentation}, 
  year={2023},
  volume={45},
  number={12},
  pages={15328-15344},
  doi={10.1109/TPAMI.2023.3319470}
}
@inproceedings{gong2021omni,
  title={Omni-supervised point cloud segmentation via gradual receptive field component reasoning},
  author={Gong, Jingyu and Xu, Jiachen and Tan, Xin and Song, Haichuan and Qu, Yanyun and Xie, Yuan and Ma, Lizhuang},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={11673--11682},
  year={2021}
}
```