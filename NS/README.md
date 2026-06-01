# FaNO on Navier-Stokes

This folder contains the minimal runnable code for FaNO on the two-dimensional incompressible Navier-Stokes benchmark, together with baseline scripts used for FNO, U-Net/ResNet, and DeepONet comparisons.

## Files

- train_ns.py: FaNO training script.
- test_ns.py: FaNO evaluation and zero-shot rollout script.
- utilities3.py: utility functions for reading .mat files and computing losses.
- baselines/eval_fno.py: FNO rollout evaluation/export script.
- baselines/train_cnn_ns_eval.py: CNN baseline training script, including U-Net and ResNet.
- baselines/export_cnn_rollout.py: CNN baseline rollout export/evaluation script.
- baselines/train_deeponet_ns_eval.py: DeepONet training/evaluation script.
- requirements.txt: minimal dependencies.

## Data

The default example assumes the Navier-Stokes data file is located at:

/path/to/NavierStokes_V1e-5_N1200_T20.mat

## Example FaNO evaluation

python test_ns.py --ckpt /path/to/fano_checkpoint.pt --test_path /path/to/NavierStokes_V1e-5_N1200_T20.mat --variant base --persistent_ratio 0.10 --modes1 8 --modes2 8 --width 20 --ntest 200 --batch_size 20 --sub 1 --S 64 --T_in 10 --T 10 --step 1 --tag fano_ns64_v1e5 --save_npz rollout_npz/fano_ns64_v1e5.npz

## Expected FaNO result

params: 190421
ZEROSHOT_RESULT tag=fano_ns64_v1e5 S=64 step=0.16141005 full=0.18895787
