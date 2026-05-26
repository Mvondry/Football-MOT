# Faster R-CNN experiments

This directory contains scripts and results for the Faster R-CNN detection experiments used in the bachelor's thesis *Football Match Analysis from Video Footage*.

The experiments are based on the SoccerNet-Tracking dataset and evaluate several Faster R-CNN variants with different class configurations and data augmentation settings.

## Directory structure

```text
Faster-R-CNN/
├── configs/                         # YAML configuration files for individual experiments
├── external/
│   └── torchvision_references/       # Modified TorchVision reference scripts
├── results/                          # Evaluation results, metric tables and plots
├── splits/                           # Train/validation/test split files
├── src/
│   ├── datasets/                     # Dataset loading code
│   ├── evaluation/                   # Evaluation scripts
│   ├── models/                       # Model definition utilities
│   ├── preprocessing/                # Annotation preprocessing scripts
│   ├── training/                     # Training scripts
│   └── visualization/                # Detection visualization scripts
├── requirements.txt
└── README.md
```

## Dataset

The dataset is not included in this repository. The scripts expect the SoccerNet-Tracking data to be available locally. The path to the dataset root directory is passed using the `--data-root` argument.

The split files in `splits/` contain paths to selected SoccerNet-Tracking sequences used for training, validation and testing.

Expected structure of each sequence:

```text
SNMOT-xxx/
├── gameinfo.ini
├── img1/
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── ...
└── gt/
    ├── gt.txt
    ├── gt_4_cls.txt
    ├── gt_5_cls.txt
    ├── gt_6_cls.txt
    └── gt_7_cls.txt
```

The original SoccerNet-Tracking `gt.txt` files do not directly contain object class labels. The class information is stored in `gameinfo.ini`, where track IDs are mapped to semantic classes.

The annotation files used for the Faster R-CNN experiments are generated during preprocessing. Bounding boxes are stored in MOT format `(x, y, w, h)` and are converted to TorchVision Faster R-CNN format `(x1, y1, x2, y2)` inside the dataset loader.

## Annotation preprocessing

Before training or evaluation, the class-specific annotation files can be generated using:

```bash
python src/preprocessing/prepare_gt_files.py \
  --data-root /path/to/SoccerNet-Tracking
```

On Windows PowerShell:

```powershell
py .\src\preprocessing\prepare_gt_files.py `
  --data-root "<path-to-SoccerNet-Tracking>"
```

The script reads the original `gt/gt.txt` file and `gameinfo.ini` in each sequence directory and creates the following files:

```text
gt/gt_7_cls.txt
gt/gt_6_cls.txt
gt/gt_5_cls.txt
gt/gt_4_cls.txt
```

The class setups are:

```text
gt_7_cls.txt:
1 = player_left
2 = goalkeeper_left
3 = player_right
4 = goalkeeper_right
5 = ball
6 = referee
7 = other

gt_6_cls.txt:
1 = player_left
2 = goalkeeper_left
3 = player_right
4 = goalkeeper_right
5 = ball
6 = referee

gt_5_cls.txt:
1 = player
2 = goalkeeper
3 = ball
4 = referee
5 = other

gt_4_cls.txt:
1 = player
2 = goalkeeper
3 = ball
4 = referee
```

The 6-class and 4-class variants remove the `other` class. The 5-class and 4-class variants also merge the left and right team classes.

For stricter preprocessing, the script can be run with:

```bash
python src/preprocessing/prepare_gt_files.py \
  --data-root /path/to/SoccerNet-Tracking \
  --strict
```

In this mode, the script raises an error if a track ID or class name cannot be mapped.

## Experiment configurations

Each experiment is defined by a YAML configuration file in `configs/`.

The following variants are included:

```text
frcnn_4_cls_noaug.yaml
frcnn_4_cls_aug.yaml
frcnn_5_cls_noaug.yaml
frcnn_5_cls_aug.yaml
frcnn_6_cls_noaug.yaml
frcnn_6_cls_aug.yaml
frcnn_7_cls_noaug.yaml
frcnn_7_cls_aug.yaml
```

The configurations differ in:

- number of object classes,
- whether the `other` class is included,
- whether team affiliation is distinguished,
- whether data augmentation is used.

The value `num_classes` in the configuration includes the background class required by TorchVision Faster R-CNN.

Class configuration overview:

```text
4 object classes + background = 5 classes
5 object classes + background = 6 classes
6 object classes + background = 7 classes
7 object classes + background = 8 classes
```

Each configuration specifies which annotation file should be used:

```text
4-class variants -> gt_4_cls.txt
5-class variants -> gt_5_cls.txt
6-class variants -> gt_6_cls.txt
7-class variants -> gt_7_cls.txt
```

Example:

```yaml
experiment_name: FRCNN-5-cls-noaug

model:
  num_classes: 6
  pretrained: true

data:
  gt_filename: gt_5_cls.txt
  class_names:
    - background
    - player
    - goalkeeper
    - ball
    - referee
    - other
```

## Installation

Create a Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

The code uses TorchVision reference detection utilities. The required files are located in:

```text
external/torchvision_references/
```

This directory should contain the modified reference scripts used during the experiments, such as:

```text
engine.py or my_engine.py
utils.py
transforms.py
coco_eval.py
coco_utils.py
```

## Training

A Faster R-CNN model can be trained using:

```bash
python src/training/train_faster_rcnn.py \
  --config configs/frcnn_5_cls_noaug.yaml \
  --data-root /path/to/SoccerNet-Tracking \
  --output-root results
```

On Windows PowerShell:

```powershell
py .\src\training\train_faster_rcnn.py `
  --config .\configs\frcnn_5_cls_noaug.yaml `
  --data-root "<path-to-SoccerNet-Tracking>" `
  --output-root results
```

To save a checkpoint after every epoch, add:

```bash
--save-all-checkpoints
```

To enable Weights & Biases logging, add:

```bash
--use-wandb
```

By default, the training script saves:

```text
results/<experiment_name>/
├── checkpoints/
│   ├── best.pth
│   └── last.pth
├── logs/
│   └── training_log.csv
└── config_used.yaml
```

The generated checkpoint files are not intended to be committed to Git.

## Evaluation

To evaluate a trained checkpoint from the local `results/` directory:

```bash
python src/evaluation/evaluate_faster_rcnn.py \
  --config configs/frcnn_5_cls_noaug.yaml \
  --checkpoint results/FRCNN-5-cls-noaug/checkpoints/best.pth \
  --data-root /path/to/SoccerNet-Tracking \
  --output-root results
```

To evaluate a pretrained checkpoint downloaded from Google Drive and placed in the root-level `model_weights/` directory:

```bash
python src/evaluation/evaluate_faster_rcnn.py \
  --config configs/frcnn_5_cls_noaug.yaml \
  --checkpoint ../model_weights/Faster-R-CNN/FRCNN-5-cls-noaug/best.pth \
  --data-root /path/to/SoccerNet-Tracking \
  --output-root results
```

On Windows PowerShell:

```powershell
py .\src\evaluation\evaluate_faster_rcnn.py `
  --config .\configs\frcnn_5_cls_noaug.yaml `
  --checkpoint ..\model_weights\Faster-R-CNN\FRCNN-5-cls-noaug\best.pth `
  --data-root "<path-to-SoccerNet-Tracking>" `
  --output-root results
```

The evaluation script computes:

- mAP,
- mAP50,
- mAP75,
- per-class precision,
- per-class recall,
- per-class F1 score,
- confusion matrix.

The outputs are saved to:

```text
results/<experiment_name>/evaluation/
├── global_metrics.txt
├── global_metrics.csv
├── confusion_matrix.csv
├── confusion_matrix.png
├── per_class_and_overall_metrics.csv
└── overall_metrics.txt
```

## Visualization

To visualize Faster R-CNN detections using a local checkpoint from `results/`:

```bash
python src/visualization/visualize_faster_rcnn.py \
  --config configs/frcnn_5_cls_noaug.yaml \
  --checkpoint results/FRCNN-5-cls-noaug/checkpoints/best.pth \
  --data-root /path/to/SoccerNet-Tracking \
  --output-root results \
  --max-sequences 3 \
  --max-frames 5 \
  --score-threshold 0.5
```

To visualize detections using a pretrained checkpoint from `model_weights/`:

```bash
python src/visualization/visualize_faster_rcnn.py \
  --config configs/frcnn_5_cls_noaug.yaml \
  --checkpoint ../model_weights/Faster-R-CNN/FRCNN-5-cls-noaug/best.pth \
  --data-root /path/to/SoccerNet-Tracking \
  --output-root results \
  --max-sequences 3 \
  --max-frames 5 \
  --score-threshold 0.5
```

On Windows PowerShell:

```powershell
py .\src\visualization\visualize_faster_rcnn.py `
  --config .\configs\frcnn_5_cls_noaug.yaml `
  --checkpoint ..\model_weights\Faster-R-CNN\FRCNN-5-cls-noaug\best.pth `
  --data-root "<path-to-SoccerNet-Tracking>" `
  --output-root results `
  --max-sequences 3 `
  --max-frames 5 `
  --score-threshold 0.5
```

Visualization outputs are saved to:

```text
results/<experiment_name>/visualizations/
```

Generated visualization images are not intended to be committed to Git.

## Model weights

Model checkpoints are not included directly in this repository due to their size.

When models are trained from scratch, checkpoints are saved into:

```text
results/<experiment_name>/checkpoints/
```

For convenience, selected pretrained checkpoints used in the thesis are provided separately via Google Drive. After downloading them, place them into the following directory structure in the root of the whole repository:

```text
model_weights/
└── Faster-R-CNN/
    ├── FRCNN-5-cls-noaug/
    │   └── best.pth
    └── FRCNN-7-cls-noaug/
        └── best.pth
```

The evaluation, visualization and tracking scripts can reference these weights using paths such as:

```text
../model_weights/Faster-R-CNN/FRCNN-5-cls-noaug/best.pth
../model_weights/Faster-R-CNN/FRCNN-7-cls-noaug/best.pth
```

Checkpoint download link:

```text
TODO: add Google Drive link
```

## Results

The `results/` directory contains exported evaluation outputs used in the thesis, including metric tables, confusion matrices and metric plots.

Generated checkpoint files and visualization images are not intended to be committed to Git.

The best Faster R-CNN variants were further used as detection models in the tracking-by-detection pipeline with DeepSORT.