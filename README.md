# ELSlowFast-LSTM
Source code of "Behavior detection of dairy goat based on YOLO11 and ELSlowFast-LSTM"

![img](https://github.com/JunpengZZhang/ELSlowFast-LSTM/figure.png)

## Requirements

The environment setup and configuration details are documented in `INSTALL.md`

## Data Preparation

To download the datasets, please refer to [DiaryGoatMVT](https://github.com/tiana-tang/DiaryGoatMVT).

We are pleased to publicly release and share the DairyGoatMVT, a multi-visual task dataset for dairy goats, which represents the culmination of over a decade of work by our team. This dataset is specifically designed to support a wide range of computer vision tasks, including object detection, object tracking, pose estimation, behavior recognition, individual identification, image generation, semantic segmentation, and instance segmentation for dairy goats. In the future, we will continue to update, refine, and expand this dataset to provide robust data support for multi-visual task models in livestock and poultry research.

You may follow the instructions in [DATASET.md](ELSlowFast-LSTM/slowfast/datasets/DATASET.md) to prepare the datasets.
## Usage

### Run command
```
python tools/run_net.py --cfg path/to/<pretrained_model_config_file>.yaml
```
