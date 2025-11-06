# GEM Dataset Setup Guide

This guide helps you set up the required datasets for training different modalities in GEM.

## Data Directory Structure

After setup, your `data/` directory should look like this:

```
data/
├── celeba/                    # CelebA images (auto-downloads)
│   └── celeba/
│       └── img_align_celeba/
├── IM-NET/                    # 3D shapes
│   └── IMSVR/
│       └── data/
│           ├── all_vox256_img_train.hdf5
│           └── all_vox256_img_test.hdf5
├── instrument.npz             # Audiovisual normalization stats
├── instrument_train.npz       # Audiovisual training data
├── instrument_test.npz        # Audiovisual test data
├── nsynth.npz                 # Audio normalization stats
├── nsynth_train.json          # NSynth training metadata
├── nsynth_test.json           # NSynth test metadata
├── nsynth_valid.json          # NSynth validation metadata
└── [audio wav files]          # NSynth audio files
```

---

## Required Data by Training Script

### 1. Image Training (CelebA)
**Script**: `train_autodecoder_multiscale.py`

**Dataset**: CelebA or CelebAHQ
- **CelebA** (recommended for beginners): Auto-downloads via torchvision
- **CelebAHQ** (1024×1024, manual): Place images in `data/celeba/`

**Setup**:
```bash
# CelebA - No setup needed! Downloads automatically on first run
python experiment_scripts/train_autodecoder_multiscale.py \
  --experiment_name celeba_test \
  --dataset celeba

# CelebAHQ - Manual download required
# Download CelebA-HQ from: https://github.com/tkarras/progressive_growing_of_gans
# Place images in: data/celeba/
```

### 2. Audio Training (NSynth)
**Script**: `train_audio_autodecoder.py`

**Required Files**:
- `nsynth.npz` - Normalization statistics
- `nsynth_train.json` - Training metadata
- `nsynth_test.json` - Test metadata
- `nsynth_valid.json` - Validation metadata
- Audio `.wav` files

**Download**:
```bash
# Download NSynth dataset from:
# https://magenta.tensorflow.org/datasets/nsynth

# Extract to data/ directory
# Ensure JSON files are in data/ and audio files are accessible
```

### 3. 3D Shape Training (IM-NET)
**Script**: `train_imnet_autodecoder.py`

**Required Files**:
- `data/IM-NET/IMSVR/data/all_vox256_img_train.hdf5` (required)
- `data/IM-NET/IMSVR/data/all_vox256_img_test.hdf5` (required)

**Download**:
```bash
# Download IM-NET dataset from:
# https://github.com/czq142857/IM-NET

# Clone the IM-NET repo and copy the data/ directory:
git clone https://github.com/czq142857/IM-NET.git
cp -r IM-NET/IMSVR/data data/IM-NET/IMSVR/

# Or download directly from the project page
```

### 4. Audiovisual Training (Instrument)
**Script**: `train_audiovisual_autodecoder.py`

**Required Files**:
- `instrument.npz` - Normalization statistics (required)
- `instrument_train.npz` - Training data
- `instrument_test.npz` - Test data

**Download**:
```bash
# Contact the original authors or check the paper supplementary materials
# Paper: https://arxiv.org/abs/2111.06387

# Place files in data/ directory
```

---

## Quick Start

### Minimal Setup (CelebA Images Only)
```bash
# No download needed! Just run:
python experiment_scripts/train_autodecoder_multiscale.py \
  --experiment_name my_first_training \
  --dataset celeba
```

The CelebA dataset will download automatically (~1.4GB).

---

## Troubleshooting

### Error: "No images found in data/celeba"
**Solution**: If using CelebAHQ, ensure images are in `data/celeba/` or specify `--dataset celeba` to auto-download CelebA.

### Error: "FileNotFoundError: instrument.npz"
**Solution**: You need to download the Instrument dataset for audiovisual training. See section 4 above.

### Error: "FileNotFoundError: all_vox256_img_train.hdf5"
**Solution**: Download the IM-NET dataset and place HDF5 files in `data/IM-NET/IMSVR/data/`.

### Error: "No such file or directory: nsynth_train.json"
**Solution**: Download the NSynth dataset and place JSON metadata files in `data/`.

---

## Dataset Sizes

| Dataset | Size | Download Time (est.) |
|---------|------|---------------------|
| CelebA (auto) | ~1.4 GB | 5-15 minutes |
| CelebA-HQ | ~90 GB | Manual download |
| NSynth | ~30 GB | Manual download |
| IM-NET | ~5 GB | Manual download |
| Instrument | ~4 GB | Contact authors |

---

## Data Configuration

### Custom Data Paths

You can specify custom data paths:

```python
# In training scripts, pass data_root parameter:
train_dataset = dataio.CelebA(sidelength=64, data_root='/my/custom/path')
```

Or set environment variable:
```bash
export GEM_DATA_ROOT=/my/custom/path
```

### Changing Default Paths

Edit `config.py`:
```python
dataset_root = '/your/custom/data/path'
log_root = '/your/custom/log/path'
```

---

## Verifying Your Setup

Run this script to check your data setup:

```bash
python -c "
import os
data_dir = 'data'

# Check what exists
datasets = {
    'CelebA (auto-download)': 'Will download on first use',
    'IM-NET': os.path.exists(os.path.join(data_dir, 'IM-NET/IMSVR/data/all_vox256_img_train.hdf5')),
    'Instrument': os.path.exists(os.path.join(data_dir, 'instrument.npz')),
    'NSynth': os.path.exists(os.path.join(data_dir, 'nsynth_train.json'))
}

print('Dataset Availability:')
for name, status in datasets.items():
    if isinstance(status, bool):
        print(f'  {name}: {\"✓ Found\" if status else \"✗ Missing\"}')
    else:
        print(f'  {name}: {status}')
"
```

---

## Additional Resources

- **Original Paper**: [Learning Signal-Agnostic Manifolds of Neural Fields](https://arxiv.org/abs/2111.06387)
- **Project Page**: Check the paper for supplementary materials
- **Data Sources**:
  - CelebA: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
  - NSynth: https://magenta.tensorflow.org/datasets/nsynth
  - IM-NET: https://github.com/czq142857/IM-NET

---

## Need Help?

If you're having trouble with data setup:
1. Check that files are in the correct directory structure
2. Verify file permissions (should be readable)
3. Check available disk space
4. Review error messages for specific missing files
5. Open an issue on GitHub with your error message
