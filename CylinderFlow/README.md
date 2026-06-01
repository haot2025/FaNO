# FaNO CylinderFlow Code

This repository contains the CylinderFlow training, evaluation, and analysis code for FaNO.

## Code structure

NORM/
  train_cflow_official_steps.py
  train_cflow_official_steps_layers_diffusion.py
  norm_dataset.py
  diffusion_net/
    layers_norm.py
    layers_diffusion.py
    layers.py
    geometry.py
    utils.py

scripts/
  plot_current_best_stepwise_allmodels.py
  export_norm_fano_pertraj_curves_oldmetric.py
  plot_cylinder_branch_variation_intensity.py

## Training

FaNO and NORM use:

PYTHONPATH=$PWD/NORM:$PWD:$PWD/scripts:$PYTHONPATH python NORM/train_cflow_official_steps.py

DiffusionNet uses:

PYTHONPATH=$PWD/NORM:$PWD:$PWD/scripts:$PYTHONPATH python NORM/train_cflow_official_steps_layers_diffusion.py

## Evaluation and analysis

Evaluation and plotting scripts are provided under scripts/.

## Data and checkpoints

Datasets, preprocessed trajectories, operator caches, checkpoints, logs, and generated plots are not included in this repository.
