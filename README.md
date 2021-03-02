# SeasonDepth
This project is for work ([arXiv](https://arxiv.org/pdf/2011.04408.pdf)) "SeasonDepth: Cross-Season Monocular Depth Prediction Dataset and Benchmark under Multiple Environments" by [Hanjiang Hu](https://hanjianghu.github.io/), Baoquan Yang, Weiang Shi, [Zhijian Qiao](https://github.com/qiaozhijian), and [Hesheng Wang](https://scholar.google.com/citations?user=q6AY9XsAAAAJ&hl=zh-CN).

[![SeasonDepth: Cross-Season Monocular Depth Prediction Dataset and Benchmark under Multiple Environments](https://res.cloudinary.com/marcomontalbano/image/upload/v1604928935/video_to_markdown/images/youtube--I2d4_wE4axs-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://youtu.be/I2d4_wE4axs "SeasonDepth: Cross-Season Monocular Depth Prediction Dataset and Benchmark under Multiple Environments")

## Test Set for Benchmark
The spilt test set of SeasonDepth is selected from the Urban slices from [CMU-Seasons](https://www.visuallocalization.net/datasets) dataset.
### Download
The images used for zero-shot experimental evaluation are available [HERE](https://drive.google.com/file/d/1UBe9K69Cjmq0m206UD2-uI4gosAJ16UA/view?usp=sharing).
### Submission
Since the benchmark website is currently under construction, please send your test results to `SeasonDepth@outlook.com`. We will process your submission in several days and give you evaluation performance.
The predicted depth map must be with `png` format in `16-bit`. The submission must be in a `zip` file with the inside folder structure below.
> results_slice2_c0
>> env00
>>> img_c0_0001.png
>>>
>>> img_c0_0002.png
>>>
>>> ...
>>
>> env00
>>
>> ...
>
> results_slice2_c0
>
> ...

## For Training and Validation Set
The training and validation set with reconstructed depth map groundtruths will be available soon.
