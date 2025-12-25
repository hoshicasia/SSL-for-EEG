# SSL for EEG

This project implements a few experiments with self-supervised learning for EEG seizure detection using FEMBA (Foundational Encoder Model with Bidirectional Mamba) on Siena-Scalp-EEG dataset as a part of final project for Self-Supervised Learning course in HSE university.

Code structure is heavily inspired by PyTorch Project Template from Deep Learning in Audio Course.

## Setup

### 0. Clone this repository

```bash
git clone https://github.com/hoshicasia/SSL-for-EEG.git
cd SSL-for-EEG
```

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Mamba (required for FEMBA)

```bash
git clone https://github.com/state-spaces/mamba.git
cd mamba
pip install . --no-build-isolation
cd ..
```

### 3. Download FEMBA Pretrained Weights

```bash
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='thorir/FEMBA', repo_type='model', local_dir='checkpoints/FEMBA')"
```

### 4. Download and Prepare Dataset

Download the Siena EEG dataset and create HDF5 file:

```bash
python scripts/download_and_prepare_data.py
```

## Experiments

### 1. Baseline: Frozen FEMBA Encoder

This experiment uses a pretrained FEMBA encoder (frozen) with a simple classifier on top.

```bash
cd /home/hoshicasia/SSL-for-EEG
python3 scripts/train_baseline_frozen_femba.py --config-name=baseline_frozen
```

### 2. Baseline with PCA

This experiment uses PCA for dimensionality reduction before classification to check if temporal features can improve model performance.


```bash
python3 scripts/train_baseline_pca.py --config-name=baseline_pca
```


### Train Frequency Encoder

This experiment trains a frequency encoder using STFT features from the EEG signals. Trained encoder is then used in the fusion experiments.

```bash
python3 scripts/train_frequency_encoder.py --config-name=frequency_encoder
```

**Note:** This step must be done previous to the following experiments as the use pretrained encoder.


### 3. Frequency Encoder with Early Fusion

This experiment trains a frequency encoder and uses early fusion for combining temporal and frequency features.

```bash
python3 scripts/train_early_fusion.py --config-name=early_fusion
```

### 4. Frequency Encoder with Late Fusion

This experiment trains a frequency encoder and uses late fusion for combining predictions from temporal and frequency branches.

```bash
python3 scripts/train_late_fusion.py --config-name=late_fusion
```
