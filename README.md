# SeasonDepth Benchmark Toolkit
This package provides a python toolkit for evaluation on the [SeasonDepth](https://seasondepth.github.io/) dataset([download](https://figshare.com/articles/dataset/SeasonDepth_Cross-Season_Monocular_Depth_Prediction_Dataset/14731323), [paper](https://arxiv.org/abs/2011.04408)). SeasonDepth is a new monocular depth prediction dataset with multi-traverse changing environments. Several representative baseline methods with recent open-source state-of-the-art pretrained models on KITTI leaderboard[[1]](#references) are evaluated on the SeasonDepth dataset without fine-tuning.
## Quick Dataset Visualization
![](figure/overview.png)
## Requirement
```shell script
pip install opencv-python
pip install xlwt xlrd tqdm
pip install numpy
```
## Evaluation
### 1. Dataset preparation
Download SeasonDepth dataset and create the directories below:
```plain
└── SeasonDepth_DATASET_ROOT
    ├── depth
    │   ├── slice2
    │   ├── slice3
    │   ├── slice7
    │   └── slice8
    ├── images
    │   ├── slice2
    │   ├── slice3
    │   ├── slice7
    │   └── slice8
```
### 2. Your results preparation
Evaluate your model on SeasonDepth and organize your results as follows:
```plain
└── YOUR_RESULT_ROOT
    ├── results2_c0
    │   ├── env00
    │   │   ├──  img_00119_c0_1303398474779439us.png
    │   │   ├──  *.png
    │   ├── env**
    ├── results2_c1
    ├── results3_c0
    ├── results3_c1
    ├── results7_c0
    ├── results7_c1
    ├── results8_c0
    └── results8_c1
```
After that, run the evaluation script in the folder `eval_code` to find your evaluation results.
```shell
python evaluation.py --pred_path YOUR_RESULT_ROOT --ground_path SeasonDepth_DATASET_ROOT
```
You can also add some following arguments if you want:
```shell
--gui # To watch the results of evaluation. Press q to exit and any other key to continue.
--disp2depth # To convert disparity map to depth map for correct evaluation.
--not_clean # To generate all the intermediate xls files during evaluating.
```

## Ealuation Results
![](figure/experiment.png)

Qualitative comparison results with illumination or vegetation changes are shown below and more can be found in [our paper](https://arxiv.org/abs/2011.04408).

![](figure/exp_visual.png)
## Cite our work
Please cite the following papers if you use our dataset:
```latex
@article{SeasonDepth,
  title={SeasonDepth: Cross-Season Monocular Depth Prediction Dataset and Benchmark under Multiple Environments},
  author={Hu, Hanjiang and Yang, Baoquan and Qiao, Zhijian and Zhao, Ding and Wang, Hesheng},
  journal={arXiv preprint arXiv:2011.04408},
  year={2021}
}
```


## References
[1] A. Geiger, P. Lenz, C. Stiller, and R. Urtasun, "Vision meets robotics: The KITTI dataset," Int. J. Robot. Research (IJRR), vol. 32, no. 11, pp. 1231–1237, Sep. 2013. [http://www.cvlibs.net/datasets/kitti/](http://www.cvlibs.net/datasets/kitti/)
