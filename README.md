
# DATE: Dual Assignment for End-to-End Fully Convolutional Object Detection

![Core](./asserts/Core.png) ![Architecture](./asserts/Arch.png)

Fully convolutional detectors discard the one-to-many assignment and adopt a one-to-one assigning strategy to achieve end-to-end detection but suffer from the slow convergence issue. In this paper, we revisit these two assignment methods and find that bringing one-to-many assignment back to end-to-end fully convolutional detectors helps with model convergence. Based on this observation, we propose **D**ual **A**ssignment for end-to-end fully convolutional de**TE**ction (**DATE**). Our method constructs two branches with one-to-many and one-to-one assignment during training and speeds up the convergence of the one-to-one assignment branch by providing more supervision signals. DATE only uses the branch with the one-to-one matching strategy for model inference, which doesn't bring inference overhead. 

## Performance

### Performance on COCO

| Model       | epoch | AP | AP50 | AP75 | APs | APm | APl | Weights | Log |
| ----------- | ----- | -- | ---- | ---- | --- | --- | --- | ------- | --- |
| DATE-R50-F  | 12    |37.3| 55.3 | 40.7 | 21.2| 40.3| 48.8| TBU     | TBU |
| DATE-R50-R  | 12    |37.0| 54.9 | 40.4 | 20.5| 39.8| 49.0| TBU     | TBU |
| DATE-R50-F  | 36    |40.6| 58.9 | 44.4 | 25.6| 44.1| 50.9| TBU     | TBU |
| DATE-R101-F | 36    |42.2| 60.6 | 46.3 | 26.6| 45.8| 54.1| TBU     | TBU |

### Performance on CrowdHuman

| Model       | iters | AP50 $\uparrow$ | mMR $\downarrow$  | Recall $\uparrow$ | Weights | Log |
| ----------- | ----- | ---- | ---- | ------ | ------- | --- |
| DATE-R50-F  | 30k   | 90.5 | 49.0 | 97.9   | TBU     | TBU |
| DATE-R50-R  | 30k   | 90.6 | 48.4 | 97.9   | TBU     | TBU |

## Installation

Our project is based on [Pytorch](https://pytorch.org/) and [mmdetection](https://github.com/open-mmlab/mmdetection/). Code is tested under Python=3.10, Pytorch>=1.12.0, mmdetection>=2.25.1.

Quick install:
```bash
git clone https://github.com/yiqunchen1999/date.git && cd date && bash ./install.sh
```

## Dataset

The dataset should be organized as following:
```
date
    |_ configs
    |_ data
        |_ coco
            |_ annotations
                |_ ...
            |_ train2017
                |_ ...
            |_ val2017
                |_ ...
            |_ ...
        |_ CrowdHuman
            |_ annotations
                |_ ...
            |_ Images
                |_ ...
```

### COCO dataset

Please follow the [tutorial of mmdetection](https://mmdetection.readthedocs.io/en/stable/1_exist_data_model.html#prepare-datasets).

### CrowdHuman

1. Download [CrowdHuman](https://www.crowdhuman.org/) to your machine;
2. Unzip and link the folder in where CrowdHuman stored to `date/data/`, i.e., 
```
date
    |_ configs
    |_ data
        |_ coco
        |_ CrowdHuman
            |_ Images
                |_ ...
            |_ annotation_train.odgt
            |_ annotation_val.odgt
            |_ ...
```
3. Run dataset converter to convert the format to COCO format:
```bash
python tools/dataset_converters/crowdhuman.py
```

## Training

To train DATE in a machine with 8 GPUs, e.g., DATE-F-R50, please run:
```bash
./tools/dist_train.sh configs/date/date_r50_12e_8x2_fcos_poto_coco.py 8
```

**NOTE:** We don't promise the code will produce the same numbers due to the randomness.

## Acknowledge

We want to thank the code of [OneNet](https://github.com/PeizeSun/OneNet) and [DeFCN](https://github.com/Megvii-BaseDetection/DeFCN). 

## LICENSE

This project is open sourced under Apache License 2.0, see [LICENSE](./LICENSE.txt).
