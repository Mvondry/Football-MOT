# Football Match Analysis from Video Footage

This repository contains the implementation and experimental results for the bachelor's thesis **Football Match Analysis from Video Footage**.

The work focuses on object detection and multi-object tracking in football match videos. The final tracking system follows the **tracking-by-detection** approach, where object detections from trained detectors are passed to the DeepSORT tracker.

The repository is divided into three main parts:

```text
Faster-R-CNN/   # Faster R-CNN detector experiments
YOLO26/         # YOLO26 detector experiments
DeepSORT/       # Tracking-by-detection pipeline and HOTA evaluation
```

## Repository structure

```text
.
├── Faster-R-CNN/
│   ├── configs/
│   ├── external/
│   ├── results/
│   ├── splits/
│   ├── src/
│   ├── requirements.txt
│   └── README.md
│
├── YOLO26/
│   ├── configs/
│   ├── results/
│   ├── splits/
│   ├── src/
│   ├── requirements.txt
│   └── README.md
│
├── DeepSORT/
│   ├── configs/
│   ├── external/
│   ├── splits/
│   ├── src/
│   ├── trackeval_data_results/
│   ├── requirements.txt
│   └── README.md
│
├── model_weights/        # Downloaded detector weights, not included in Git
└── README.md
```

Each subdirectory contains its own detailed README with instructions for preprocessing, training, evaluation and visualization.

## Project overview

The repository contains scripts for:

- preparing SoccerNet-Tracking annotations,
- training Faster R-CNN detection models,
- training YOLO26 detection models,
- evaluating object detectors,
- visualizing detection outputs,
- running DeepSORT tracking using trained detectors,
- evaluating tracking results using the HOTA metric,
- visualizing tracking results.

The following pipelines are implemented:

```text
Faster R-CNN → detection evaluation
YOLO26       → detection evaluation

Faster R-CNN → DeepSORT → HOTA evaluation
YOLO26       → DeepSORT → HOTA evaluation
```

## Dataset

The experiments use the SoccerNet-Tracking dataset.

The dataset is not included in this repository.

It must be downloaded separately from the official SoccerNet source and placed locally.

Dataset link:

```text
https://www.soccer-net.org/data
```

The expected sequence structure is:

```text
SNMOT-xxx/
├── img1/
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── ...
├── gt/
│   └── gt.txt
└── seqinfo.ini
```

## Model weights

Model checkpoints are not included directly in this repository due to their size.

When models are trained from scratch, the training scripts save checkpoints into detector-specific `results/` directories:

```text
YOLO26/results/<experiment_name>/weights/best.pt
Faster-R-CNN/results/<experiment_name>/checkpoints/best.pth
```

These files are not committed to Git.

For convenience, the pretrained detector weights used in the thesis are provided separately via Google Drive. After downloading them, place them into the root-level `model_weights/` directory:

```text
model_weights/
├── YOLO26/
│   ├── YOLO26s-aug/
│   │   └── best.pt
│   └── YOLO26l-aug/
│       └── best.pt
└── Faster-R-CNN/
    ├── FRCNN-5-cls-noaug/
    │   └── best.pth
    └── FRCNN-7-cls-noaug/
        └── best.pth
```

The DeepSORT configs reference these downloaded detector weights using relative paths such as:

```text
../model_weights/YOLO26/YOLO26l-aug/best.pt
../model_weights/Faster-R-CNN/FRCNN-5-cls-noaug/best.pth
```

If you train the detector models yourself and want to use your newly trained checkpoints for tracking, either copy them into `model_weights/` using the structure above, or update the `detector.weights` path in the corresponding DeepSORT YAML config.

The DeepSORT re-identification checkpoint `ckpt.t7` is handled separately. It should be placed inside the cloned `deep_sort_pytorch` repository:

```text
DeepSORT/external/deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7
```

Checkpoint download link:

```text
https://drive.google.com/drive/folders/1yFcThZ3A7n93yMePAcFdbyxFzkby4oMc?usp=sharing
```

## Faster R-CNN experiments

The `Faster-R-CNN/` directory contains scripts for training, evaluating and visualizing Faster R-CNN detection models.

Main components:

```text
Faster-R-CNN/
├── configs/                 # experiment configs
├── external/                # TorchVision reference scripts
├── results/                 # text results, metrics and plots
├── splits/                  # train/val/test split files
└── src/
    ├── datasets/            # dataset loading
    ├── evaluation/          # detector evaluation
    ├── models/              # Faster R-CNN model definition
    ├── preprocessing/       # annotation preprocessing
    ├── training/            # training script
    └── visualization/       # detection visualization
```

Typical setup:

```bash
cd Faster-R-CNN
pip install -r requirements.txt
```

Training:

```bash
python src/training/train_faster_rcnn.py \
  --config configs/frcnn_5_cls_noaug.yaml \
  --data-root /path/to/SoccerNet-Tracking \
  --output-root results
```

Evaluation using downloaded weights:

```bash
python src/evaluation/evaluate_faster_rcnn.py \
  --config configs/frcnn_5_cls_noaug.yaml \
  --checkpoint ../model_weights/Faster-R-CNN/FRCNN-5-cls-noaug/best.pth \
  --data-root /path/to/SoccerNet-Tracking \
  --output-root results
```

Visualization using downloaded weights:

```bash
python src/visualization/visualize_faster_rcnn.py \
  --config configs/frcnn_5_cls_noaug.yaml \
  --checkpoint ../model_weights/Faster-R-CNN/FRCNN-5-cls-noaug/best.pth \
  --data-root /path/to/SoccerNet-Tracking \
  --output-root results \
  --max-sequences 1 \
  --max-frames 10
```

More detailed instructions are available in:

```text
Faster-R-CNN/README.md
```

## YOLO26 experiments

The `YOLO26/` directory contains scripts for converting SoccerNet-Tracking annotations to YOLO format, training YOLO26 models, evaluating them and visualizing predictions.

Main components:

```text
YOLO26/
├── configs/                 # experiment configs
├── results/                 # text results, metrics and plots
├── splits/                  # train/val/test split files
└── src/
    ├── preprocessing/       # SoccerNet/MOT to YOLO conversion
    ├── training/            # YOLO26 training
    ├── evaluation/          # YOLO26 evaluation
    └── visualization/       # detection visualization
```

Typical setup:

```bash
cd YOLO26
pip install -r requirements.txt
```

Dataset conversion:

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

Training:

```bash
python src/training/train_yolo.py \
  --config configs/yolo26s_aug.yaml \
  --output-root results
```

Evaluation using downloaded weights:

```bash
python src/evaluation/evaluate_yolo.py \
  --config configs/yolo26s_aug.yaml \
  --weights ../model_weights/YOLO26/YOLO26s-aug/best.pt \
  --output-root results \
  --split test
```

Visualization using downloaded weights:

```bash
python src/visualization/visualize_yolo.py \
  --config configs/yolo26s_aug.yaml \
  --weights ../model_weights/YOLO26/YOLO26s-aug/best.pt \
  --output-root results \
  --split test \
  --max-images 10
```

More detailed instructions are available in:

```text
YOLO26/README.md
```

## DeepSORT tracking pipeline

The `DeepSORT/` directory contains the final tracking-by-detection pipeline.

It connects trained detectors with the DeepSORT tracker and evaluates the resulting trajectories using HOTA through `sn-trackeval`.

Main components:

```text
DeepSORT/
├── configs/                         # tracking configs
├── external/                        # external repositories, not included in Git
├── splits/                          # split files
├── src/
│   ├── common/                      # shared utilities
│   ├── detectors/                   # YOLO and Faster R-CNN detector wrappers
│   ├── preprocessing/               # TrackEval GT preparation
│   ├── tracking/                    # detector + DeepSORT scripts
│   ├── evaluation/                  # HOTA evaluation
│   └── visualization/               # tracking visualization
└── trackeval_data_results/          # text-based tracking results
```

The tracking pipeline depends on two external repositories:

```text
deep_sort_pytorch
sn-trackeval
```

Clone them into `DeepSORT/external/`:

```bash
cd DeepSORT
mkdir external
cd external

git clone https://github.com/ZQPei/deep_sort_pytorch.git
git clone https://github.com/SoccerNet/sn-trackeval.git

cd ..
```

Install requirements:

```bash
pip install -r requirements.txt
```

The DeepSORT re-identification checkpoint `ckpt.t7` should be placed inside the cloned `deep_sort_pytorch` repository:

```text
DeepSORT/external/deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7
```

Prepare TrackEval-compatible ground truth:

```bash
python src/preprocessing/prepare_trackeval_gt.py \
  --data-root /path/to/SoccerNet-Tracking \
  --split-file splits/test.txt \
  --output-root trackeval_data_results \
  --gt-filename gt.txt \
  --overwrite
```

Run YOLO26l-aug + DeepSORT:

```bash
python src/tracking/run_yolo_deepsort.py \
  --config configs/yolo26l_aug_deepsort.yaml
```

Run Faster R-CNN 5-class noaug + DeepSORT:

```bash
python src/tracking/run_frcnn_deepsort.py \
  --config configs/frcnn_5_cls_noaug_deepsort.yaml
```

Evaluate HOTA:

```bash
python src/evaluation/evaluate_hota.py \
  --config configs/yolo26l_aug_deepsort.yaml \
  --split test
```

Visualize tracking results:

```bash
python src/visualization/visualize_tracks.py \
  --config configs/yolo26l_aug_deepsort.yaml \
  --max-sequences 1 \
  --max-frames 50 \
  --save-video
```

More detailed instructions are available in:

```text
DeepSORT/README.md
```

## Results

Text-based experimental results are included in the corresponding `results/` or `trackeval_data_results/` directories.

Typical included result files:

```text
*.txt
*.csv
*.json
*.yaml
confusion_matrix.png
metric plots
```

Large generated files are not intended to be committed, including:

```text
model checkpoints
datasets
external repositories
detection/tracking visualization frames
videos
temporary debug outputs
```

## Reproducibility notes

The repository is organized so that the experiments can be reproduced in three stages:

1. Train and evaluate object detectors.
2. Use trained detectors in the DeepSORT tracking pipeline.
3. Evaluate tracking results using HOTA.

Recommended order:

```text
1. Faster-R-CNN preprocessing/training/evaluation
2. YOLO26 dataset conversion/training/evaluation
3. DeepSORT tracking and HOTA evaluation
```

Debug configuration files are included for quick testing of individual scripts. Debug runs are intended only to verify that the code works and should not be used for final metric reporting.

In particular, tracking debug runs using `--max-frames` should not be used for final HOTA values, because TrackEval compares tracker outputs against the full ground-truth sequence.

## GitHub storage policy

The repository is intended to contain:

```text
source code
configuration files
README files
split files
text-based results
metric tables
selected plots such as confusion matrices
```

The repository is not intended to contain:

```text
datasets
virtual environments
model checkpoints
external repositories
large detection caches
generated visualization frames
videos
```

Model checkpoints are provided separately via Google Drive.

## Author

Martin Vondráček

Bachelor's thesis project focused on football match analysis from video footage using object detection and multi-object tracking.
