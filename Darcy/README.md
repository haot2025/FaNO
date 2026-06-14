# FaNO

Minimal training code for FaNO and baseline models on the Darcy flow benchmark.

## Included models

- FaNO
- FNO
- U-Net
- LSM

## Environment

Tested with Python 3.10 and PyTorch.

```bash
pip install -r requirements.txt
```

## Data

Place the standard FNO Darcy flow benchmark files under `data/`:

```text
data/piececonst_r421_N1024_smooth1.mat
data/piececonst_r421_N1024_smooth2.mat
```

The default setting uses 1,000 training samples, 100 test samples, and an `85 x 85` subsampled grid.

## Train FaNO

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_darcy_fano.py
```

Default configuration:

```text
modes = 12
width = 64
persistent_ratio = 0.2
ntrain = 1000
ntest = 100
batch_size = 20
epochs = 500
learning_rate = 0.001
weight_decay = 0.0001
```

The main FaNO classes are:

```text
conv_fano
FaNO_block
FaNO_Net
```

## Train FNO

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_darcy_fno.py
```

## Train U-Net

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_darcy_unet.py --width 64 --modes 12 --ntrain 1000 --ntest 100 --batch_size 20 --epochs 500 --lr 0.001
```

## Train LSM

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_darcy_lsm.py --ntrain 1000 --ntest 100 --r 5 --s 85 --epochs 500 --batch_size 20 --d_model 64 --num_token 4 --num_basis 12 --patch_size 3,3 --padding 11,11 --lr 0.001 --weight_decay 0.0001 --tag darcy_lsm_r421_N1000_ep500_m12_w64_bs20
```

## Results

| Model | Parameters | Test relative L2 error |
|---|---:|---:|
| FNO | 9.504M | 1.018e-02 |
| U-Net | 31.036M | 6.844e-03 |
| LSM | 19.187M | 7.118e-03 |
| FaNO | 7.621M | 5.619e-03 |

For models with complex-valued Fourier weights, each complex parameter is counted as two real-valued parameters.

## Output

Training scripts write checkpoints, predictions, and metrics to `model/`, `pred/`, `results/`, and `logs/`.

## Repository structure

```text
FaNO_github/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── scripts/
│   ├── train_darcy_fano.py
│   ├── train_darcy_fno.py
│   ├── train_darcy_unet.py
│   └── train_darcy_lsm.py
└── src/
    ├── __init__.py
    └── data/
        ├── __init__.py
        └── utilities3.py
```

Datasets, checkpoints, predictions, and logs are not included.
