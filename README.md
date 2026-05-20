# Point Transformer PartNet Baseline

This repository now contains a runnable baseline pipeline:

- H5/PLY preprocessing to normalized sampled NPZ files
- Dataset loader for processed samples
- Lightweight point-transformer-style segmentation model
- Training and evaluation scripts with checkpoints and metrics
- Basic tests for dataset and metrics

## 1) Install

```bash
pip install -r requirements.txt
```

## 2) Preprocess

```bash
python tools/preprocess_partnet.py
```

Outputs:

- Processed samples: `datasets/processed/partnet_pt/samples/...`
- Splits: `datasets/splits/train.txt`, `datasets/splits/val.txt`, `datasets/splits/test.txt`
- Meta: `datasets/processed/partnet_pt/meta/stats.json`

## 3) Train

```bash
python train.py --config configs/partnet_pt_baseline.yaml
```

Outputs:

- Logs: `experiments/exp_pt_partnet_baseline/train_log.txt`
- Checkpoints: `experiments/exp_pt_partnet_baseline/checkpoints/`

## 4) Evaluate

```bash
python eval.py --config configs/partnet_pt_baseline.yaml --checkpoint experiments/exp_pt_partnet_baseline/checkpoints/best.pth --split test --save-viz
```

Outputs:

- Predictions: `outputs/preds/*.npz`
- Visualization: `outputs/viz/*_pred.ply`

## 5) Run tests

```bash
pytest -q
```
>>>>>>> f2e5b59 (Initial commit)
