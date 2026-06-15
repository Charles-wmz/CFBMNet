# Smartphone Microphone-Based Pulmonary Function Estimation Using a Curve-First Basis-Mixture Network

This repository contains the core implementation of a Curve-First Basis-Mixture Network (CFBMNet) for estimating pulmonary function from smartphone microphone recordings. The model predicts a full expiratory flow curve first, then derives clinically relevant spirometric indices such as FEV1, FVC, PEF, and FEV1/FVC from the predicted curve.

No raw audio, processed data, trained weights, or experimental result files are included in this repository.

## Overview

CFBMNet uses Mel spectrograms extracted from forced-expiration audio as the primary input. Optional demographic variables can be fused through an enhanced demographic encoder. Instead of directly regressing only scalar pulmonary function indices, the network predicts a 60-point flow-time curve over a 3-second window.

The main components are:

- A convolutional condition encoder for Mel spectrograms.
- An optional enhanced demographic encoder with feature interaction and self-attention.
- A basis-mixture curve decoder that learns reusable curve bases and condition-specific mixture coefficients.
- Physics-informed training losses based on curve reconstruction, FEV1, FVC, PEF, and curve smoothness.
- Subject-independent 5-fold cross-validation training and evaluation.

## Repository Structure

```text
.
├── config.py                  # Default paths, model dimensions, and training hyperparameters
├── preprocess_data.py          # Audio and flow-curve preprocessing pipeline
├── train_CV.py                 # 5-fold cross-validation training and testing script
├── trainer.py                  # PyTorch training loop and uncertainty-weighted losses
├── models/
│   ├── condition_encoder.py
│   ├── demographic_encoder.py
│   ├── enhanced_demographic_encoder.py
│   ├── model_registry.py
│   └── regression_model.py
├── utils/
│   ├── dataset.py
│   ├── losses.py
│   └── metrics.py
└── requirements.txt
```

## Data Preparation

The code expects processed data under `data_pre/` by default:

```text
data_pre/
├── mel/        # Mel spectrogram files, one .npy file per sample
├── csv/        # Processed flow curves, one .csv file per sample
├── meta/       # Optional metadata generated during preprocessing
└── label.csv   # Subject-level labels and demographic variables
```

The expected label file includes subject identifiers and pulmonary function targets. The dataset loader supports fields such as `id`, `FEV1`, `FVC`, `PEF`, `gender`, `age`, `high` or `height`, and `weight`.

To preprocess data from local audio and flow-curve files:

```bash
python preprocess_data.py --wav_dir ./data_temporary/wav --csv_dir ./data_temporary/csv --output_dir ./data_pre
```

## Installation

```bash
pip install -r requirements.txt
```

PyTorch should be installed with the CUDA version appropriate for your local system if GPU training is required.

## Training

Run subject-independent 5-fold cross-validation:

```bash
python train_CV.py --exp cfbmnet
```

By default, training enables the basis-mixture curve decoder, enhanced demographic encoder, SpecAugment, and physics-informed smooth loss with uncertainty-weight warmup.

Common options:

```bash
python train_CV.py \
  --exp cfbmnet \
  --use-basis-mixture-curve-decoder true \
  --basis-mixture-num-bases 8 \
  --use-enhanced-demographic true \
  --demographic-features gender,height,weight \
  --use-phys-smooth-warm true
```

Outputs are written to `cv_results/<exp>/`, including fold checkpoints, logs, configuration snapshots, aggregate metrics, and sample-level prediction files.

## Notes

This repository intentionally excludes baseline and comparison model implementations. It is intended to provide the clean core code for the CFBMNet method described in the project title.

Before publishing, add an appropriate license file if the code will be released publicly.
