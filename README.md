# Seeing Through Co-occurrence

Official implementation of **Seeing Through Co-occurrence: Multi-dimensional
Decoupling for Multi-label Temporal Action Detection**.

This repository provides the Charades training pipeline used in the paper.
Only pre-computed RGB features are required; videos and features are not
redistributed.

## Repository structure

```text
train_charades.py                 # training entry point
model/
  multidimensional_decoupling.py  # complete model
  temporal_encoder.py             # Temporal Encoder
  relation_encoder.py             # Relation Encoder
  two_stream_mixer.py             # Two-Stream Mixer
  classification_head.py          # Classification Head
datasets/charades.py              # Charades loader
data/charades.json                # Charades annotations
utils.py, apmeter.py              # training utilities and metrics
```

## Installation

```bash
python -m pip install -r requirements.txt
```

## Data preparation

Download Charades from the official provider and extract one feature file per
video. Set `-rgb_root` to that directory. Each NumPy array must have shape
`(T, D)`, where `D=1024` for I3D RGB or `D=768` for CLIP RGB features.

## Training

```bash
python train_charades.py -rgb_root /path/to/charades_features
```

The default settings are 256 clips, batch size 32, learning rate `1e-4`, and
50 epochs. Use `-gpu 0` to select a GPU and `-save_root` to choose the output
directory. The text-guided auxiliary branch uses
`sentence-transformers/all-MiniLM-L6-v2` and downloads it on first use.

## Citation

Please cite the accompanying paper when using this code.
