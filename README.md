# UESTC4006P YOLO26 Experiments

## 1. Overview
This repository is used for the graduation project **"Design and Implementation of Road Crack Detection System Based on YOLO Network Model"**, focusing on YOLO26-related detection and segmentation experiments.

## 2. Repository Role
This repository is dedicated to YOLO26-related experiments for road crack detection and segmentation.  
It is maintained separately from the main project repository to avoid affecting the main YOLOv8/YOLOv11 experimental pipeline.

## 3. Relationship with Other Repositories
- Main project repository:  
  https://github.com/ScatteredWood/UESTC4006P-Individual-Project/tree/feature/segmentation-improvement
- GUI repository:  
  https://github.com/ScatteredWood/uestc4006p-gui
- YOLO26 repository:  
  Current repository (`ultralytics_yolo26`), used only for YOLO26 experiments and comparison.

## 4. Project Structure
The following structure reflects the current repository content:

```text
ultralytics_yolo26/
|- ultralytics/                          # Ultralytics/YOLO26 core framework code
|- docs/                                 # Documentation sources
|- examples/                             # Example code from upstream
|- tests/                                # Test suite
|- docker/                               # Docker-related files
|- batch_eval_yolo26.py                  # Batch validation/export script for YOLO26 runs
|- batch_predict_det_models_yolo26.py    # Batch detection prediction script
|- batch_predict_seg_models_yolo26.py    # Batch segmentation prediction script
|- batch_cascade_pairs_yolo26.py         # Batch cascade det+seg script
|- pyproject.toml                        # Build/dependency configuration
`- README.md
```

## 5. Environment
- Python requirement (from `pyproject.toml`): `>=3.8`
- Core dependencies include `torch`, `torchvision`, `opencv-python`, `pyyaml`, `numpy`, etc.
- CLI entrypoint: `yolo` (defined by `project.scripts` in `pyproject.toml`)

Reference environment:

> The experiments were mainly conducted on Ubuntu 22.04 with CUDA 12.8 and PyTorch 2.8.0. Please adapt the environment according to the available GPU and CUDA version.

Typical setup:

```bash
pip install -e .
```

## 6. Dataset Preparation
Datasets are not included in this repository.  
Please prepare your own dataset and dataset YAML files (for example, `data/crack_det.yaml` or `data/crack_seg.yaml`) and pass their paths through CLI arguments or environment variables.

Do not hardcode personal absolute paths in reusable scripts.  
If example commands contain local paths, replace them with your own environment paths.

## 7. Training
Example training commands (YOLO26 detection/segmentation):

```bash
# Detection
yolo task=detect mode=train model=yolo26n.yaml data=path/to/crack_det.yaml imgsz=1024 epochs=100 batch=8

# Segmentation
yolo task=segment mode=train model=yolo26n-seg.yaml data=path/to/crack_seg.yaml imgsz=1024 epochs=100 batch=8
```

## 8. Validation
Validation can be executed by YOLO CLI or by the batch validation script in this repository:

```powershell
# Single model validation (CLI)
yolo task=detect mode=val model=path/to/best.pt data=path/to/crack_det.yaml imgsz=1024

# Batch validation/export for YOLO26 runs
$env:YOLO26_RUNS_ROOT="path/to/runs_root"
$env:YOLO26_DET_VAL_IMAGES="path/to/det/images/val"
$env:YOLO26_SEG_VAL_IMAGES="path/to/seg/images/val"
python batch_eval_yolo26.py
```

Validation outputs are typically written to `runs/` or the configured export/report directories.

## 9. Prediction / Inference
Prediction examples:

```powershell
# Detection prediction (CLI)
yolo task=detect mode=predict model=path/to/best.pt source=path/to/images

# Batch prediction scripts
$env:YOLO26_RUNS_ROOT="path/to/runs_root"
$env:YOLO26_DET_SOURCE_DIR="path/to/det_test_images"
python batch_predict_det_models_yolo26.py

$env:YOLO26_SEG_SOURCE_DIR="path/to/seg_test_images"
python batch_predict_seg_models_yolo26.py

$env:YOLO26_CASCADE_SOURCE_DIR="path/to/test_images"
$env:YOLO26_CASCADE_SCRIPT="path/to/cascade_infer_detseg.py"
python batch_cascade_pairs_yolo26.py
```

Prediction visualisations are saved to configured output directories.

## 10. Experimental Outputs
Experimental outputs such as training logs, validation curves, prediction visualisations and model weights may be stored locally under directories such as `runs/`, `logs/`, `outputs/`, `results/` or `checkpoints/`. These files are treated as experimental artefacts and must not be removed during repository cleanup.

## 11. Notes
- This repository is not the GUI application.
- This repository is not the main YOLOv8/YOLOv11 improvement repository.
- It is maintained separately for YOLO26 experiments and comparison.

## 12. License / Acknowledgement
This repository inherits the upstream license file currently present in this repository: `LICENSE` (AGPL-3.0).  
Upstream project reference: https://github.com/ultralytics/ultralytics
