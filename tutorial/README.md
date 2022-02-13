# Tutorial for the SeasonDepth Prediction Challenge

## Access to the dataset

Follow the link of [training set v1.1 download](https://figshare.com/articles/dataset/SeasonDepth_Cross-Season_Monocular_Depth_Prediction_Training_Dataset/16442025) and [validation set download](https://figshare.com/articles/dataset/SeasonDepth_Cross-Season_Monocular_Depth_Prediction_Dataset/14731323). Refer to our [website](https://seasondepth.github.io/) and [benchmark toolkit](https://github.com/SeasonDepth/SeasonDepth) for more details about the data structure.

## Demo for Supervised Learning Track (SLT): [BTS](https://arxiv.org/abs/1907.10326)

- Download training set v1.1 from the above link and move it to `TRAINING_PATH = /path/to/dataset/training_set`. The structure of folders is highly recommended to be organized like below:

    ```plain
    └── TRAINING_PATH
        ├── rgb
        │   ├── img_xxxxx_c0_xxxxxxxxxxxxxxxx.jpg
        │   └── ...
        ├── depth
        │   ├── img_xxxxx_c0_xxxxxxxxxxxxxxxx.png
        │   └── ...
    ```

- Download the code from [bts](https://github.com/cleinc/bts). We are going to use its pytorch implement. 
- Generate the train_list from `TRAINING_PATH`. Each line of the list should consist of paths of an rgb image, its ground truth and focal length of the camera. Camera intrinsics con be found from [intrinsics.txt](https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/Extended-CMU-Seasons/intrinsics.txt). An example is shown below:

    `$TRAINING_PATH/rgb/img_xxxxx_c0_xxxxxxxxxxxxxxxx.jpg $TRAINING_PATH/depth/img_xxxxx_c0_xxxxxxxxxxxxxxxx.png focal_c0`

Considering that ground truth of Seasondepth is relative range, it can be aligned in all kind of methods, so focal length is not that important while fine-tuning.

- Start training. First, you are suggested to modify the `__getiterm__` method  in bts_dataloader.py according to your path and format of `train_list.txt`. Second, you should align the ground truth with predicted depth maps before computing the loss, with either average value or other sota methods. Finally, you will be able to start training with the model and hyperparameters pre-trained on KITTI dataset. Here we will show how to align with average value. 

    ```python
    ...
        lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est = model(image, focal)

        depth_gt = depth_gt.data.cpu().numpy()
        depth_est_np = depth_est.data.cpu().numpy()
        depth_gt = depth_gt * depth_est_np[depth_gt > 0].mean() / depth_gt[depth_gt > 0].mean()
        depth_gt[depth_gt < 0] = 0
        depth_gt = torch.autograd.Variable(torch.from_numpy(depth_gt).cuda(args.gpu, non_blocking=True))

        if args.dataset == 'nyu':
            mask = depth_gt > 0.1
        else:
            mask = depth_gt > 1.0

        loss = silog_criterion.forward(depth_est, depth_gt, mask.to(torch.bool))
    ...
    ```

## Demo for Self-Supervised Learning Track (SSLT): [SfMLearner](https://arxiv.org/abs/1704.07813)

- Download training set v1.1 from the above link and move it to `TRAINING_PATH = /path/to/dataset/training_set`. Ground truth is not necessery while training, but we must ensure that images in a folder is several consecutive frames captured by one single camera. Below is an example of organizing the sequences:

    ```plain
    └── TRAINING_PATH
        ├── slice6_env00_c0
        │   ├── img_xxxxx_c0_xxxxxxxxxxxxxxxx.jpg
        │   └── ...
        ├── slice6_env00_c1
        │   ├── img_xxxxx_c0_xxxxxxxxxxxxxxxx.png
        │   └── ...
        ├── slice6_env01_c0
        │   ├── img_xxxxx_c1_xxxxxxxxxxxxxxxx.png
        │   └── ...
        ├── ...
    ```

- Preprocessing of the dataset. According to [issue#108](https://github.com/ClementPinard/SfmLearner-Pytorch/issues/108), we shuold create train.txt and val.txt files where folder (each containes a video sequence in jpeg pictures) is used either for training or validation. According to [issue#125](https://github.com/ClementPinard/SfmLearner-Pytorch/issues/125#issuecomment-859009283), the model pretrained on kitti only works well with images of size 416x128, so we resized the rgb images to 416x128 to make training results better.
- Start training. Clone the repo from [SfmLearner-Pytorch](https://github.com/ClementPinard/SfmLearner-Pytorch) and then start training with the pretrained model and arguments: [Pretrained Nets](https://github.com/ClementPinard/SfmLearner-Pytorch#pretrained-nets).

## Reference

    [1] Lee, J. H., Han, M. K., Ko, D. W., & Suh, I. H. (2019). From big to small: Multi-scale local planar guidance for monocular depth estimation. arXiv preprint arXiv:1907.10326.

    [2] Zhou, T., Brown, M., Snavely, N., & Lowe, D. G. (2017). Unsupervised learning of depth and ego-motion from video. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1851-1858).

    [3] Hu, H., Yang, B., Qiao, Z., Zhao, D., & Wang, H. (2021). SeasonDepth: Cross-Season Monocular Depth Prediction Dataset and Benchmark under Multiple Environments. arXiv preprint arXiv:2011.04408.