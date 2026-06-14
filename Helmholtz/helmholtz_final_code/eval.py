import os
import argparse
import random
import logging
import numpy as np
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt
import wandb

from utils import logging_utils
logging_utils.config_logger()

from utils.YParams import YParams
from utils.inferencer import Inferencer
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ruamelDict


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_field_figure(source, gt, pred, save_path, title_prefix="sample"):
    err = np.abs(pred - gt)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    im0 = axes[0, 0].imshow(source, origin="lower")
    axes[0, 0].set_title(f"{title_prefix} | source")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(gt, origin="lower")
    axes[0, 1].set_title(f"{title_prefix} | GT")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[1, 0].imshow(pred, origin="lower")
    axes[1, 0].set_title(f"{title_prefix} | Pred")
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im3 = axes[1, 1].imshow(err, origin="lower")
    axes[1, 1].set_title(f"{title_prefix} | Abs Error")
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_line_figure(gt, pred, save_path, title_prefix="sample"):
    h, w = gt.shape
    row = h // 2
    col = w // 2

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(gt[row, :], label="GT")
    axes[0].plot(pred[row, :], label="Pred")
    axes[0].set_title(f"{title_prefix} | Center Row")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(gt[:, col], label="GT")
    axes[1].plot(pred[:, col], label="Pred")
    axes[1].set_title(f"{title_prefix} | Center Col")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_spectrum_figure(gt, pred, save_path, title_prefix="sample"):
    gt_ft = np.fft.rfft2(gt)
    pred_ft = np.fft.rfft2(pred)
    err_ft = np.fft.rfft2(pred - gt)

    gt_mag = np.log1p(np.abs(gt_ft))
    pred_mag = np.log1p(np.abs(pred_ft))
    err_mag = np.log1p(np.abs(err_ft))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    im0 = axes[0].imshow(gt_mag, origin="lower", aspect="auto")
    axes[0].set_title(f"{title_prefix} | log|FFT(GT)|")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(pred_mag, origin="lower", aspect="auto")
    axes[1].set_title(f"{title_prefix} | log|FFT(Pred)|")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(err_mag, origin="lower", aspect="auto")
    axes[2].set_title(f"{title_prefix} | log|FFT(Error)|")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_visualization_from_inferencer(inferencer, args):
    if inferencer.world_rank != 0:
        return

    save_dir = args.save_dir
    ensure_dir(save_dir)

    fields = getattr(inferencer, "vis_fields", None)
    if fields is None or len(fields) == 0:
        logging.warning("No vis_fields found on inferencer, skip visualization.")
        return

    num_triplets = len(fields) // 3

    with open(os.path.join(save_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"model: {inferencer.params.model}\n")
        f.write(f"weights: {inferencer.params.weights}\n")
        f.write(f"num_visualized_samples: {num_triplets}\n")
        for k, v in inferencer.logs.items():
            f.write(f"{k}: {v}\n")

    for i in range(num_triplets):
        source = fields[3 * i + 0]
        gt = fields[3 * i + 1]
        pred = fields[3 * i + 2]

        rel = np.linalg.norm(pred - gt) / (np.linalg.norm(gt) + 1e-12)
        prefix = f"sample_{i:02d}_relL2_{rel:.6f}"

        save_field_figure(
            source, gt, pred,
            os.path.join(save_dir, prefix + "_fields.png"),
            title_prefix=prefix
        )
        save_line_figure(
            gt, pred,
            os.path.join(save_dir, prefix + "_lines.png"),
            title_prefix=prefix
        )
        save_spectrum_figure(
            gt, pred,
            os.path.join(save_dir, prefix + "_spectrum.png"),
            title_prefix=prefix
        )

    logging.info(f"Visualization saved to {save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", default='./config/operators.yaml', type=str)
    parser.add_argument("--config", default='default', type=str)
    parser.add_argument("--root_dir", default='./', type=str, help='root dir to store results')
    parser.add_argument("--run_num", default='0', type=str, help='sub run config')
    parser.add_argument("--sweep", default='none', type=str)
    parser.add_argument("--weights", default='./ckpt.tar', type=str)

    parser.add_argument("--save_dir", default='./vis', type=str)
    parser.add_argument("--num_vis", default=6, type=int)
    parser.add_argument("--sample_mode", default='best', type=str, choices=['best', 'worst', 'random'])
    parser.add_argument("--seed", default=0, type=int)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    params = YParams(os.path.abspath(args.yaml_config), args.config)
    logging.info("Starting config {}".format(args.config))

    params['weights'] = args.weights
    params['num_vis'] = args.num_vis

    if hasattr(params, 'weights'):
        logging.info("with weights {}".format(params.weights))
    else:
        raise RuntimeError("no model weights provided")

    inferencer = Inferencer(params, args)

    if inferencer.world_rank == 0:
        hparams = ruamelDict()
        yaml = YAML()
        for key, value in params.params.items():
            hparams[str(key)] = str(value)
        with open(os.path.join(params['experiment_dir'], 'hyperparams.yaml'), 'w') as hpfile:
            yaml.dump(hparams, hpfile)

    inferencer.launch()

    if dist.is_initialized():
        dist.barrier()

    run_visualization_from_inferencer(inferencer, args)

    if dist.is_initialized():
        dist.barrier()

    logging.info("Finished config {}".format(args.config))
    logging.info('DONE')
