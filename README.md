# SSL for EEG

This project implements a few experiments with self-supervised learning for EEG seizure detection using FEMBA (Foundational Encoder Model with Bidirectional Mamba) on Siena-Scalp-EEG dataset.

In order to run experiments:

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
