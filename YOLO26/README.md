# YOLO26 experiments

This directory contains scripts and results for the YOLO26 detection experiments used in the bachelor's thesis *Football Match Analysis from Video Footage*.

The YOLO26 models were trained for the 4-class detection setup:

```text
0 = player
1 = goalkeeper
2 = ball
3 = referee
```

The experiments compare different YOLO26 model sizes and the effect of additional data augmentation.

## Directory structure

```text
YOLO26/
├── configs/                         # YAML configuration files for YOLO26 experiments
├── results/                         # Evaluation results, metric tables and plots
├── splits/                          # Train/validation/test split files
├── src/
│   ├── preprocessing/               # Conversion from SoccerNet/MOT format to YOLO format
│   ├── training/                    # YOLO26 training scripts
│   ├── evaluation/                  # Evaluation scripts
│   └── visualization/               # Detection visualization scripts
├── requirements.txt
└── README.md
```

## Dataset

The original SoccerNet-Tracking dataset is not included in this repository. The scripts expect the data to be available locally.

The original tracking annotations are stored in MOT-like format. For YOLO training, the data must first be converted to the standard YOLO format.

The expected SoccerNet-Tracking sequence structure is:

```text
SNMOT-xxx/
├── img1/
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── ...
└── gt/
    └── gt_4_cls.txt
```

The `gt_4_cls.txt` files are generated during the preprocessing step in the Faster R-CNN part of the repository. They contain four object classes:

```text
1 = player
2 = goalkeeper
3 = ball
4 = referee
```

During conversion to YOLO format, these labels are remapped to zero-based class IDs:

```text
0 = player
1 = goalkeeper
2 = ball
3 = referee
```

## YOLO dataset conversion

To convert the SoccerNet-Tracking data to YOLO format, run:

```bash
python src/preprocessing/convert_snmot_to_yolo.py \
  --data-root /path/to/SoccerNet-Tracking \
  --train-split splits/train.txt \
  --val-split splits/val.txt \
  --test-split splits/test.txt \
  --gt-filename gt_4_cls.txt \
  --output-root /path/to/yolo_snmot_4cls \
  --num-classes 4 \
  --class-names "player,goalkeeper,ball,referee" \
  --transfer-mode copy
```

On Windows PowerShell:

```powershell
py .\src\preprocessing\convert_snmot_to_yolo.py `
  --data-root "<path-to-SoccerNet-Tracking>" `
  --train-split .\splits\train.txt `
  --val-split .\splits\val.txt `
  --test-split .\splits\test.txt `
  --gt-filename gt_4_cls.txt `
  --output-root "<path-to-yolo_snmot_4cls>" `
  --num-classes 4 `
  --class-names "player,goalkeeper,ball,referee" `
  --transfer-mode copy
```

For a quick debug conversion, use:

```powershell
py .\src\preprocessing\convert_snmot_to_yolo.py `
  --data-root "<path-to-SoccerNet-Tracking>" `
  --train-split .\splits\debug_train.txt `
  --val-split .\splits\debug_val.txt `
  --test-split .\splits\debug_test.txt `
  --gt-filename gt_4_cls.txt `
  --output-root "<path-to-yolo_snmot_4cls_debug>" `
  --num-classes 4 `
  --class-names "player,goalkeeper,ball,referee" `
  --transfer-mode copy `
  --max-sequences 1 `
  --max-frames 10
```

The output YOLO dataset has the following structure:

```text
yolo_snmot_4cls/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── snmot.yaml
```

The generated `snmot.yaml` file is used by Ultralytics during training and evaluation.

## Experiment configurations

Each experiment is defined by a YAML file in `configs/`.

The main experiment variants are:

```text
yolo26s_noaug.yaml
yolo26s_aug.yaml
yolo26m_noaug.yaml
yolo26m_aug.yaml
yolo26l_noaug.yaml
yolo26l_aug.yaml
```

The variants differ in:

- YOLO26 model size,
- use of additional data augmentation.

The `noaug` variants do not use the additional augmentation settings defined in this repository. Ultralytics may still apply its own default training transformations unless explicitly disabled.

Example configuration:

```yaml
experiment_name: YOLO26s-aug

model: yolo26s.pt

data: /path/to/yolo_snmot_4cls/snmot.yaml

training:
  epochs: 100
  imgsz: 640
  batch: 16
  device: 0
  workers: 6
  seed: 0
  patience: 50
  exist_ok: true
  plots: true

evaluation:
  imgsz: 640
  batch: 16
  device: 0
  workers: 2

augmentation:
  enabled: true
  mixup: 0.15
  cutmix: 0.10
  degrees: 3.0
  shear: 2.0
  perspective: 0.0005
  multi_scale: 0.25

  albumentations:
    blur: 0.05
    median_blur: 0.02
    to_gray: 0.02
    clahe: 0.05
    random_brightness_contrast: 0.20
    random_gamma: 0.15
    image_compression: 0.20
```

## Installation

Create a Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

## Training

To train a YOLO26 model, run:

```bash
python src/training/train_yolo.py \
  --config configs/yolo26s_aug.yaml \
  --output-root results
```

On Windows PowerShell:

```powershell
py .\src\training\train_yolo.py `
  --config .\configs\yolo26s_aug.yaml `
  --output-root results
```

To enable Weights & Biases logging, add:

```bash
--use-wandb
```

Example debug run:

```powershell
py .\src\training\train_yolo.py `
  --config .\configs\debug_yolo26s_noaug.yaml `
  --output-root results
```

Training outputs are saved to:

```text
results/<experiment_name>/
├── weights/
│   ├── best.pt
│   └── last.pt
├── args.yaml
├── results.csv
└── ...
```

The generated checkpoint files are not intended to be committed to Git.

## Evaluation

To evaluate a checkpoint from the local `results/` directory:

```bash
python src/evaluation/evaluate_yolo.py \
  --config configs/yolo26s_aug.yaml \
  --weights results/YOLO26s-aug/weights/best.pt \
  --output-root results \
  --split test
```

To evaluate a pretrained checkpoint downloaded from Google Drive and placed in the root-level `model_weights/` directory:

```bash
python src/evaluation/evaluate_yolo.py \
  --config configs/yolo26s_aug.yaml \
  --weights ../model_weights/YOLO26/YOLO26s-aug/best.pt \
  --output-root results \
  --split test
```

On Windows PowerShell:

```powershell
py .\src\evaluation\evaluate_yolo.py `
  --config .\configs\yolo26s_aug.yaml `
  --weights ..\model_weights\YOLO26\YOLO26s-aug\best.pt `
  --output-root results `
  --split test `
  --plots `
  --exist-ok
```

The evaluation script computes:

- precision,
- recall,
- mAP50,
- mAP75,
- mAP50-95,
- per-class metrics,
- inference speed.

The outputs are saved to:

```text
results/<experiment_name>/evaluation/
├── metrics_summary.txt
├── metrics_summary.json
├── per_class_metrics.csv
├── per_class_metrics.json
├── ultralytics_summary.csv
├── ultralytics_summary.json
└── ultralytics_outputs/
```

## Visualization

To visualize detections using a checkpoint from the local `results/` directory:

```bash
python src/visualization/visualize_yolo.py \
  --config configs/yolo26s_aug.yaml \
  --weights results/YOLO26s-aug/weights/best.pt \
  --output-root results \
  --split test \
  --max-images 10 \
  --conf 0.25
```

To visualize detections using a pretrained checkpoint from `model_weights/`:

```bash
python src/visualization/visualize_yolo.py \
  --config configs/yolo26s_aug.yaml \
  --weights ../model_weights/YOLO26/YOLO26s-aug/best.pt \
  --output-root results \
  --split test \
  --max-images 10 \
  --conf 0.25
```

On Windows PowerShell:

```powershell
py .\src\visualization\visualize_yolo.py `
  --config .\configs\yolo26s_aug.yaml `
  --weights ..\model_weights\YOLO26\YOLO26s-aug\best.pt `
  --output-root results `
  --split test `
  --max-images 10 `
  --conf 0.25
```

To visualize detections on a specific image or directory:

```powershell
py .\src\visualization\visualize_yolo.py `
  --config .\configs\yolo26s_aug.yaml `
  --weights ..\model_weights\YOLO26\YOLO26s-aug\best.pt `
  --source "<path-to-images>" `
  --output-root results `
  --max-images 10 `
  --conf 0.25
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
results/<experiment_name>/weights/
```

For convenience, selected pretrained checkpoints used in the thesis are provided separately via Google Drive. After downloading them, place them into the following directory structure in the root of the whole repository:

```text
model_weights/
└── YOLO26/
    ├── YOLO26s-aug/
    │   └── best.pt
    └── YOLO26l-aug/
        └── best.pt
```

The evaluation, visualization and tracking scripts can reference these weights using paths such as:

```text
../model_weights/YOLO26/YOLO26s-aug/best.pt
../model_weights/YOLO26/YOLO26l-aug/best.pt
```

Checkpoint download link:

```text
TODO: add Google Drive link
```

## Results

The `results/` directory contains exported evaluation outputs used in the thesis, including metric tables, JSON summaries and Ultralytics metric plots.

Generated checkpoint files and visualization images are not intended to be committed to Git.

The best YOLO26 variants were further used as detection models in the tracking-by-detection pipeline with DeepSORT.