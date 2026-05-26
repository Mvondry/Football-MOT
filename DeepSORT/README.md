# DeepSORT tracking experiments

This directory contains the tracking-by-detection part of the bachelor's thesis *Football Match Analysis from Video Footage*.

The tracking pipeline connects trained object detectors with the DeepSORT tracker and evaluates the resulting trajectories using the HOTA metric through `sn-trackeval`.

The following detector variants are supported:

```text
YOLO26s-aug        → DeepSORT → HOTA evaluation
YOLO26l-aug        → DeepSORT → HOTA evaluation
FRCNN-5-cls-noaug  → DeepSORT → HOTA evaluation
FRCNN-7-cls-noaug  → DeepSORT → HOTA evaluation
```

## Directory structure

```text
DeepSORT/
├── configs/                         # YAML configuration files for tracking experiments
├── external/                        # External repositories, not included in Git
│   ├── deep_sort_pytorch/            # ZQPei/deep_sort_pytorch
│   └── sn-trackeval/                 # SoccerNet/sn-trackeval
├── splits/                          # Train/validation/test split files
├── src/
│   ├── common/                       # Shared utility functions
│   ├── detectors/                    # Detector wrappers for YOLO and Faster R-CNN
│   ├── evaluation/                   # HOTA evaluation through sn-trackeval
│   ├── preprocessing/                # Preparation of TrackEval-compatible GT data
│   ├── tracking/                     # YOLO/Faster R-CNN + DeepSORT tracking scripts
│   └── visualization/                # Visualization of tracking outputs
├── trackeval_data_results/           # Generated TrackEval-compatible text results
├── requirements.txt
└── README.md
```

## External dependencies

This part depends on two external repositories:

```text
deep_sort_pytorch
sn-trackeval
```

They are not included directly in this repository. Clone them into the `external/` directory:

```bash
mkdir external
cd external

git clone https://github.com/ZQPei/deep_sort_pytorch.git
git clone https://github.com/SoccerNet/sn-trackeval.git

cd ..
```

The external directory should then look like this:

```text
DeepSORT/
└── external/
    ├── deep_sort_pytorch/
    └── sn-trackeval/
```

The DeepSORT re-identification checkpoint `ckpt.t7` is also not included in this repository. It should be placed inside the cloned `deep_sort_pytorch` repository:

```text
DeepSORT/external/deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7
```

If the checkpoint is stored elsewhere, update the path in the corresponding YAML config file.

## Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

The external repositories must be available in `external/`, or their paths must be changed in the config files:

```yaml
external:
  deep_sort_pytorch: external/deep_sort_pytorch
  sn_trackeval: external/sn-trackeval
```

## Model weights

Detector weights are not included directly in this repository due to their size.

When models are trained from scratch, the detector training scripts save checkpoints into their own `results/` directories:

```text
YOLO26/results/<experiment_name>/weights/best.pt
Faster-R-CNN/results/<experiment_name>/checkpoints/best.pth
```

For convenience, the pretrained detector weights used in the thesis are provided separately via Google Drive. After downloading them, place them into the following directory structure in the root of the whole repository:

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

Since the tracking scripts are run from the `DeepSORT/` directory, the config files reference these weights using relative paths such as:

```text
../model_weights/YOLO26/YOLO26l-aug/best.pt
../model_weights/Faster-R-CNN/FRCNN-5-cls-noaug/best.pth
```

If you train detector models yourself and want to use your own checkpoints for tracking, either copy them into `model_weights/` using the structure above, or update the `detector.weights` path in the corresponding DeepSORT YAML config.

The DeepSORT checkpoint remains inside the external `deep_sort_pytorch` repository:

```text
external/deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7
```

## Dataset

The original SoccerNet-Tracking dataset is not included in this repository.

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

The split files in `splits/` should contain one sequence per line. The path can be either absolute or relative to the dataset root, for example:

```text
test/SNMOT-116
test/SNMOT-123
```

or directly:

```text
SNMOT-116
SNMOT-123
```

If only sequence names are used, the scripts search for the matching sequence directory under the dataset root. If multiple directories with the same sequence name exist, use more specific paths such as `train/SNMOT-xxx` or `test/SNMOT-xxx`.

## TrackEval data preparation

Before HOTA evaluation, the ground-truth data must be converted into a flat TrackEval-compatible structure.

Run:

```bash
python src/preprocessing/prepare_trackeval_gt.py \
  --data-root /path/to/SoccerNet-Tracking \
  --split-file splits/test.txt \
  --output-root trackeval_data_results \
  --gt-filename gt.txt \
  --overwrite
```

On Windows PowerShell:

```powershell
py .\src\preprocessing\prepare_trackeval_gt.py `
  --data-root "<path-to-SoccerNet-Tracking>" `
  --split-file .\splits\test.txt `
  --output-root trackeval_data_results `
  --gt-filename gt.txt `
  --overwrite
```

This creates:

```text
trackeval_data_results/
├── gt/
│   ├── SNMOT-116/
│   │   ├── seqinfo.ini
│   │   └── gt/
│   │       └── gt.txt
│   └── ...
└── seqmaps/
    └── test_seqmap.txt
```

Images are not copied into `trackeval_data_results`, because TrackEval only needs ground-truth files, tracker output files and a sequence map.

For HOTA tracking evaluation, the original `gt.txt` files are used.

## YOLO26 + DeepSORT tracking

The YOLO26 tracking pipeline is implemented in:

```text
src/tracking/run_yolo_deepsort.py
```

Run YOLO26s-aug + DeepSORT:

```bash
python src/tracking/run_yolo_deepsort.py \
  --config configs/yolo26s_aug_deepsort.yaml
```

On Windows PowerShell:

```powershell
py .\src\tracking\run_yolo_deepsort.py `
  --config .\configs\yolo26s_aug_deepsort.yaml
```

Run YOLO26l-aug + DeepSORT:

```bash
python src/tracking/run_yolo_deepsort.py \
  --config configs/yolo26l_aug_deepsort.yaml
```

On Windows PowerShell:

```powershell
py .\src\tracking\run_yolo_deepsort.py `
  --config .\configs\yolo26l_aug_deepsort.yaml
```

For a quick debug run:

```powershell
py .\src\tracking\run_yolo_deepsort.py `
  --config .\configs\debug_yolo_deepsort.yaml `
  --max-sequences 1 `
  --max-frames 20
```

## Faster R-CNN + DeepSORT tracking

The Faster R-CNN tracking pipeline is implemented in:

```text
src/tracking/run_frcnn_deepsort.py
```

Run FRCNN-5-cls-noaug + DeepSORT:

```bash
python src/tracking/run_frcnn_deepsort.py \
  --config configs/frcnn_5_cls_noaug_deepsort.yaml
```

On Windows PowerShell:

```powershell
py .\src\tracking\run_frcnn_deepsort.py `
  --config .\configs\frcnn_5_cls_noaug_deepsort.yaml
```

Run FRCNN-7-cls-noaug + DeepSORT:

```bash
python src/tracking/run_frcnn_deepsort.py \
  --config configs/frcnn_7_cls_noaug_deepsort.yaml
```

On Windows PowerShell:

```powershell
py .\src\tracking\run_frcnn_deepsort.py `
  --config .\configs\frcnn_7_cls_noaug_deepsort.yaml
```

For a quick debug run:

```powershell
py .\src\tracking\run_frcnn_deepsort.py `
  --config .\configs\debug_frcnn_deepsort.yaml `
  --max-sequences 1 `
  --max-frames 20
```

## Tracking output format

The tracking output is saved into:

```text
trackeval_data_results/
└── trackers/
    └── <tracker_name>/
        ├── data/
        │   └── SNMOT-xxx.txt
        ├── data_with_classes/
        │   └── SNMOT-xxx.txt
        └── timing_summary.csv
```

The `data/` folder contains standard MOT-format files for TrackEval:

```text
frame,id,bb_left,bb_top,bb_width,bb_height,conf,-1,-1,-1
```

The `data_with_classes/` folder contains an additional class ID in the last column and is used only for visualization:

```text
frame,id,bb_left,bb_top,bb_width,bb_height,conf,-1,-1,-1,class_id
```

The `data/` and `data_with_classes/` folders can be large and are not intended to be committed to Git.

## HOTA evaluation

HOTA evaluation is implemented in:

```text
src/evaluation/evaluate_hota.py
```

Run HOTA evaluation for YOLO26s-aug + DeepSORT:

```bash
python src/evaluation/evaluate_hota.py \
  --config configs/yolo26s_aug_deepsort.yaml \
  --split test
```

For YOLO26l-aug + DeepSORT:

```bash
python src/evaluation/evaluate_hota.py \
  --config configs/yolo26l_aug_deepsort.yaml \
  --split test
```

For FRCNN-5-cls-noaug + DeepSORT:

```bash
python src/evaluation/evaluate_hota.py \
  --config configs/frcnn_5_cls_noaug_deepsort.yaml \
  --split test
```

For FRCNN-7-cls-noaug + DeepSORT:

```bash
python src/evaluation/evaluate_hota.py \
  --config configs/frcnn_7_cls_noaug_deepsort.yaml \
  --split test
```

On Windows PowerShell:

```powershell
py .\src\evaluation\evaluate_hota.py `
  --config .\configs\yolo26l_aug_deepsort.yaml `
  --split test
```

The specific sequences used for evaluation are determined by the seqmap file:

```yaml
trackeval:
  output_root: trackeval_data_results
  tracker_name: YOLO26l-aug-DeepSORT
  seqmap_name: test_seqmap.txt
```

The `--split` argument sets the split name used internally by TrackEval, but the actual evaluated sequences are defined by the selected seqmap file.

## TrackEval class name

The `MotChallenge2DBox` wrapper in TrackEval internally uses the class name `pedestrian`. Therefore, the internal evaluation class is kept as:

```yaml
trackeval_class: pedestrian
```

For clearer output names, an alias can be used:

```yaml
output_class_name: football
```

This creates additional files such as:

```text
football_summary.txt
football_summary.csv
```

while keeping TrackEval internally compatible.

## Visualization

Tracking visualization is implemented in:

```text
src/visualization/visualize_tracks.py
```

It reads files from:

```text
trackeval_data_results/trackers/<tracker_name>/data_with_classes/
```

and draws tracking results on the original frames.

Run visualization for YOLO26l-aug + DeepSORT:

```bash
python src/visualization/visualize_tracks.py \
  --config configs/yolo26l_aug_deepsort.yaml \
  --max-sequences 1 \
  --max-frames 50 \
  --save-video
```

On Windows PowerShell:

```powershell
py .\src\visualization\visualize_tracks.py `
  --config .\configs\yolo26l_aug_deepsort.yaml `
  --max-sequences 1 `
  --max-frames 50 `
  --save-video
```

For Faster R-CNN:

```powershell
py .\src\visualization\visualize_tracks.py `
  --config .\configs\frcnn_5_cls_noaug_deepsort.yaml `
  --max-sequences 1 `
  --max-frames 50 `
  --save-video
```

Visualization outputs are saved to:

```text
trackeval_data_results/
└── trackers/
    └── <tracker_name>/
        └── visualizations/
            ├── SNMOT-xxx/
            │   ├── 000001.jpg
            │   ├── 000002.jpg
            │   └── ...
            └── SNMOT-xxx.mp4
```

Generated visualization images and videos are not intended to be committed to Git.

## Class ID offsets for visualization

YOLO26 uses zero-based class IDs:

```text
0 = player
1 = goalkeeper
2 = ball
3 = referee
```

Therefore, YOLO tracking configs should contain:

```yaml
visualization:
  track_subdir: data_with_classes
  class_id_offset: 0
  draw_confidence: false
  video_fps: 25
  class_names:
    - player
    - goalkeeper
    - ball
    - referee
```

Faster R-CNN uses one-based class labels because label `0` is reserved for background.

For FRCNN-5-cls-noaug:

```text
1 = player
2 = goalkeeper
3 = ball
4 = referee
5 = other
```

The corresponding visualization config is:

```yaml
visualization:
  track_subdir: data_with_classes
  class_id_offset: 1
  draw_confidence: false
  video_fps: 25
  class_names:
    - player
    - goalkeeper
    - ball
    - referee
    - other
```

For FRCNN-7-cls-noaug:

```text
1 = player_left
2 = goalkeeper_left
3 = player_right
4 = goalkeeper_right
5 = ball
6 = referee
7 = other
```

The corresponding visualization config is:

```yaml
visualization:
  track_subdir: data_with_classes
  class_id_offset: 1
  draw_confidence: false
  video_fps: 25
  class_names:
    - player_left
    - goalkeeper_left
    - player_right
    - goalkeeper_right
    - ball
    - referee
    - other
```

## Example YOLO26s-aug DeepSORT config

```yaml
experiment_name: YOLO26s-aug-DeepSORT

external:
  deep_sort_pytorch: external/deep_sort_pytorch
  sn_trackeval: external/sn-trackeval

data:
  data_root: /path/to/SoccerNet-Tracking
  split_file: splits/test.txt

detector:
  weights: ../model_weights/YOLO26/YOLO26s-aug/best.pt
  image_size: 640
  confidence_threshold: 0.35
  device: 0
  keep_classes:

deepsort:
  checkpoint: external/deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7
  params:
    max_dist: 0.1
    min_confidence: 0.0
    nms_max_overlap: 1.0
    max_iou_distance: 0.85
    max_age: 40
    n_init: 4
    nn_budget: 100

trackeval:
  output_root: trackeval_data_results
  tracker_name: YOLO26s-aug-DeepSORT
  seqmap_name: test_seqmap.txt
  split: test
  trackeval_class: pedestrian
  output_class_name: football

output:
  save_class_output: true
  class_match_iou_threshold: 0.05
  debug: false

visualization:
  track_subdir: data_with_classes
  class_id_offset: 0
  draw_confidence: false
  video_fps: 25
  class_names:
    - player
    - goalkeeper
    - ball
    - referee
```

## Example YOLO26l-aug DeepSORT config

```yaml
experiment_name: YOLO26l-aug-DeepSORT

external:
  deep_sort_pytorch: external/deep_sort_pytorch
  sn_trackeval: external/sn-trackeval

data:
  data_root: /path/to/SoccerNet-Tracking
  split_file: splits/test.txt

detector:
  weights: ../model_weights/YOLO26/YOLO26l-aug/best.pt
  image_size: 640
  confidence_threshold: 0.35
  device: 0
  keep_classes:

deepsort:
  checkpoint: external/deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7
  params:
    max_dist: 0.1
    min_confidence: 0.0
    nms_max_overlap: 1.0
    max_iou_distance: 0.85
    max_age: 40
    n_init: 4
    nn_budget: 100

trackeval:
  output_root: trackeval_data_results
  tracker_name: YOLO26l-aug-DeepSORT
  seqmap_name: test_seqmap.txt
  split: test
  trackeval_class: pedestrian
  output_class_name: football

output:
  save_class_output: true
  class_match_iou_threshold: 0.05
  debug: false

visualization:
  track_subdir: data_with_classes
  class_id_offset: 0
  draw_confidence: false
  video_fps: 25
  class_names:
    - player
    - goalkeeper
    - ball
    - referee
```

## Example FRCNN-5-cls-noaug DeepSORT config

```yaml
experiment_name: FRCNN-5-cls-noaug-DeepSORT

external:
  deep_sort_pytorch: external/deep_sort_pytorch
  sn_trackeval: external/sn-trackeval

data:
  data_root: /path/to/SoccerNet-Tracking
  split_file: splits/test.txt

detector:
  weights: ../model_weights/Faster-R-CNN/FRCNN-5-cls-noaug/best.pth
  num_classes: 6
  confidence_threshold: 0.75
  device: cpu
  keep_labels:

deepsort:
  checkpoint: external/deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7
  params:
    max_dist: 0.1
    min_confidence: 0.0
    nms_max_overlap: 1.0
    max_iou_distance: 0.95
    max_age: 40
    n_init: 7
    nn_budget: 100

trackeval:
  output_root: trackeval_data_results
  tracker_name: FRCNN-5-cls-noaug-DeepSORT
  seqmap_name: test_seqmap.txt
  split: test
  trackeval_class: pedestrian
  output_class_name: football

output:
  save_class_output: true
  class_match_iou_threshold: 0.05
  debug: false

visualization:
  track_subdir: data_with_classes
  class_id_offset: 1
  draw_confidence: false
  video_fps: 25
  class_names:
    - player
    - goalkeeper
    - ball
    - referee
    - other
```

## Example FRCNN-7-cls-noaug DeepSORT config

```yaml
experiment_name: FRCNN-7-cls-noaug-DeepSORT

external:
  deep_sort_pytorch: external/deep_sort_pytorch
  sn_trackeval: external/sn-trackeval

data:
  data_root: /path/to/SoccerNet-Tracking
  split_file: splits/test.txt

detector:
  weights: ../model_weights/Faster-R-CNN/FRCNN-7-cls-noaug/best.pth
  num_classes: 8
  confidence_threshold: 0.75
  device: cpu
  keep_labels:

deepsort:
  checkpoint: external/deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7
  params:
    max_dist: 0.1
    min_confidence: 0.0
    nms_max_overlap: 1.0
    max_iou_distance: 0.95
    max_age: 40
    n_init: 7
    nn_budget: 100

trackeval:
  output_root: trackeval_data_results
  tracker_name: FRCNN-7-cls-noaug-DeepSORT
  seqmap_name: test_seqmap.txt
  split: test
  trackeval_class: pedestrian
  output_class_name: football

output:
  save_class_output: true
  class_match_iou_threshold: 0.05
  debug: false

visualization:
  track_subdir: data_with_classes
  class_id_offset: 1
  draw_confidence: false
  video_fps: 25
  class_names:
    - player_left
    - goalkeeper_left
    - player_right
    - goalkeeper_right
    - ball
    - referee
    - other
```

## Results

Text-based results are stored under:

```text
trackeval_data_results/trackers/<tracker_name>/
```

Typical result files include:

```text
timing_summary.csv
pedestrian_summary.txt
pedestrian_summary.csv
football_summary.txt
football_summary.csv
```

Large tracking files, images and videos are not intended to be committed to Git.

## Notes

- Debug runs using `--max-frames` should not be used for final HOTA values, because TrackEval compares the tracker output against the full ground-truth sequence.
- If only the first few frames are processed, HOTA will be artificially low due to many false negatives.
- The `data_with_classes/` files are used only for visualization, not for HOTA evaluation.
- HOTA evaluation uses standard MOT-format files from the `data/` directory.