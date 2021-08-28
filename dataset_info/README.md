# SeasonDepth Test and Training Set for Benchmark

This is the public release of the SeasonDepth test and training set that are used in this paper to 
benchmark monocular depth prediction algorithms under changing environments.

This dataset is based on the CMU Visual Localization dataset described here:
```
Hernán Badino, Daniel Huber, and Takeo Kanade. 
Visual topometric localization. 
In 2011 IEEE Intelligent Vehicles Symposium (IV), pages 794–799. IEEE, 2011.
```
This SeasonDepth  dataset also refers to CMU-Seasons dataset here:
```
T. Sattler, W. Maddern, C. Toft, A. Torii, L. Hammarstrand, E. Stenborg, D. Safari, M. Okutomi, M. Pollefeys, J. Sivic, F. Kahl, T. Pajdla. 
Benchmarking 6DOF Outdoor Visual Localization in Changing Conditions. 
Conference on Computer Vision and Pattern Recognition (CVPR) 2018 
```

The test set of SeasonDepth dataset basically includes images from *slice2, slice3, slice7, slice8* and training set for version 1.1 includes *slice4，slice5，slice6* under urban area in CMU-Seasons dataset. The depth maps are reconstructed through SfM algorithm with carefully mannual refinement as groundtruths to evaulate the robustness of depth prediction models from KITTI leaderboard under multiple environmental conditions.

> The format of test set can be found [here](https://github.com/SeasonDepth/SeasonDepth/blob/master/README.md#evaluation). The training set is organized as below:
```plain
└── SeasonDepth_trainingset_v1.1
    ├── slice4
    │   ├── env00
    		├── c0
    			├── images
    			├── depth_maps
    		├── c1
    			├── images
    			├── depth_maps
    │   ├── env01
    │   ├── ...
    │   └── env11
    ├── slice5
    ├── slice6
    ├── README.md
    ├── intrinsics.txt
```

## License
SeasonDepth dataset is built on CMU Visual Localization dataset created by 
Hernan Badino, Daniel Huber, and Takeo Kanade and also with a reference of CMU-Seasons dataset derived by Torsten Sattler et al.
As the previous works are licensed under 
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 License](https://creativecommons.org/licenses/by-nc-sa/4.0/) from the README file [here](https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/Extended-CMU-Seasons/), SeasonDepth is also licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 License](https://creativecommons.org/licenses/by-nc-sa/4.0/).


Please see this [webpage](https://www.visuallocalization.net/) and  if you are
interested in using the images commercially.

### Using the Extended CMU Seasons Dataset
By using the SeasonDepth dataset, you agree to the license terms set out above.
If you are using theSeasonDepth dataset in a publication, please cite **all** of the
following three sources:
```
@article{hu2020seasondepth,
  title={SeasonDepth: Cross-Season Monocular Depth Prediction Dataset and Benchmark under Multiple Environments},
  author={Hu, Hanjiang and Yang, Baoquan and Qiao, Zhijian and Zhao, Ding and Wang, Hesheng},
  journal={arXiv preprint arXiv:2011.04408},
  year={2021}
}

@inproceedings{Sattler2018CVPR,
  author={Sattler, Torsten and Maddern, Will and Toft, Carl and Torii, Akihiko and Hammarstrand, Lars and Stenborg, Erik and Safari, Daniel and Okutomi, Masatoshi and Pollefeys, Marc and Sivic, Josef and Kahl, Fredrik and Pajdla, Tomas},
  title={{Benchmarking 6DOF Outdoor Visual Localization in Changing Conditions}},
  booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2018},
}

@inproceedings{badino2011visual,
  title={Visual topometric localization},
  author={Badino, Hern{\'a}n and Huber, Daniel and Kanade, Takeo},
  booktitle={2011 IEEE Intelligent Vehicles Symposium (IV)},
  pages={794--799},
  year={2011},
  organization={IEEE}
}
```


### Privacy
We take privacy seriously. If you have any concerns regarding the images and other data
provided with this dataset, please [contact us](mailto:hanjianghu@cmu.edu).



## Provided Files
The following files are provided with this release of the SeasonDepth dataset in the root directory.
* `intrinsics.txt`: Contains the intrinsic calibrations of the two cameras used in the dataset from `intrinsics.txt` [here](https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/Extended-CMU-Seasons/intrinsics.txt). Both the RGB images and depth maps have been undistorted according to the intrinsics.
* `RGB.zip`: Contains all RGB images from `env00` to `env11` ended with `.jpg`. The `c0` subfolder inside each environment indicates the images are captured from the camera 0 and `c1` is from camera 1.
* `depth.zip`: Contains all depth images from `env00` to `env11` ended with `.png`. The `c0` subfolder inside each environment indicates the images are captured from the camera 0 and `c1` is from camera 1. The file name of depth map is identical to that of corresponding RGB image and the depth value is stored in `uint16`.

### Image Details
The SeasonDepth dataset uses images captured by left and right cameras mounted on a car. 
We provide a list of the 
different conditions with their corresponding capture dates, the folder indexes of environments used in the SeasonDepth dataset and first 5 digits of timestamps in the name of image files and depth map files.


Condition | Capture Date | Folder Index | First 5 Digits of Timestamp
------------|---------------- | ---------------| --------------- 
Sunny + No Foliage | 4 Apr 2011             | env00 | 13033
Sunny + Foliage | 1 Sep 2010                | env01 | 12833
Sunny + Foliage | 15 Sep 2010               | env02 | 12845
Cloudy + Foliage | 1 Oct 2010               | env03 | 12859
Sunny + Foliage | 19 Oct 2010               | env04 | 12875
Overcast + Mixed Foliage | 28 Oct 2010      | env05 | 12881
Low Sun + Mixed Foliage | 3 Nov 2010        | env06 | 12887
Low Sun + Mixed Foliage | 12 Nov 2010       | env07 | 12895
Cloudy + Mixed Foliage | 22 Nov 2010        | env08 | 12904
Low Sun + No Foliage + Snow | 21 Dec 2010   | env09 | 12929
Low Sun + Foliage | 4 Mar 2011              | env10 | 12992
Overcast + Foliage | 28 Jul 2011            | env11 | 13118

### Explanation of the File Name
Take the name of `img_00122_c1_1283347879534213us.jpg` for RGB file as example, `img_00122` indicates that this image is the *122th* image in the dataset. The infix `_c1_` indicates that camera 1 was used to capture the image. `1283347879534213us` is the timestamp of the capture time.

### Use for Benchmark
The details about the instruction for the benchmark can be found on the [toolkit repo](https://github.com/SeasonDepth/SeasonDepth).


## Reference
1. Torsten Sattler, Will Maddern, Carl Toft, Akihiko Torii, Lars Hammarstrand, Erik Stenborg, Daniel Safari, Masatoshi Okutomi, Marc Pollefeys, Josef Sivic, et al. Benchmarking 6dof outdoor visual localization in changing conditions. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 8601–8610, 2018. https://www.visuallocalization.net/.
2. Aayush Bansal, Hernan Badino, and Daniel Huber. Understanding how camera configuration and environ337
mental conditions affect appearance-based localization. In IEEE Intelligent Vehicles (IV), 2014.
3. Hernán Badino, Daniel Huber, and Takeo Kanade. Visual topometric localization. In 2011 IEEE Intelligent Vehicles Symposium (IV), pages 794–799. IEEE, 2011.
4. Torsten Sattler *et al.*, `README.md`. https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/Extended-CMU-Seasons/, 2018.
