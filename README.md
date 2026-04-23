<h1 align="center">
<br>
OrderDP: A Theoretically Guaranteed Lossless Dynamic Data Pruning Framework
</h1>

<p align="center"><b>ICLR 2026 Poster</b> | <a href="https://openreview.net/forum?id=e77QyyRQPz">[Paper]</a></p>

<p align="center">
  <strong>A plug-and-play dynamic data pruning framework with theoretical guarantees for lossless training acceleration.</strong>
</p>

## Overview

Data pruning aims to reduce training cost by discarding samples that appear less informative, while preserving the accuracy of full-dataset training. Existing approaches often rely on heuristic importance scores, which can introduce biased gradient estimation and make their optimization behavior hard to characterize.

OrderDP addresses this issue with a simple two-stage strategy: it first samples a random subset of the training set, then keeps the top-q samples within that subset according to the current surrogate loss. This design yields a practical dynamic pruning rule with theoretical guarantees and strong empirical performance.

<p align="center">
  <img src="assets/overview.png" width="100%">
</p>

This repository contains the current PyTorch code release used for our CIFAR and ImageNet experiments, including:

- full-data baselines,
- InfoBatch comparison code used in the paper,
- OrderDP implementations for CIFAR and ImageNet.

## Repository Structure

```text
OrderDP/
├── assets/
├── cifar/
│   ├── cifar_example.py
│   ├── orderDP_example.py
│   ├── model.py
│   ├── lars.py
│   ├── lamb.py
│   ├── infobatch/
│   └── orderDP/
└── imagenet/
    ├── prune_experiment_orderdp.py
    ├── prune_experiment_unsup.py
    ├── orderdp_dataloader.py
    ├── infobatch_dataloader.py
    ├── lars.py
    ├── r50_orderdp_90epoch.sh
    └── r50_unsup_90.sh
```

## Environment

The codebase assumes a standard PyTorch training environment with:

- Python 3.9+
- PyTorch
- torchvision
- numpy
- matplotlib

Install the missing dependencies in your own environment before running experiments.

## CIFAR Experiments

Run commands from the repository root:

```bash
python3 cifar/cifar_example.py \
  --model r50 --dataset cifar100 --optimizer sgd --max-lr 0.03 \
  --batch-size 128 --num_epoch 200
```

InfoBatch baseline:

```bash
python3 cifar/cifar_example.py \
  --model r50 --dataset cifar100 --optimizer sgd --max-lr 0.03 \
  --ratio 0.5 --batch-size 128 --num_epoch 200 \
  --is_anealing 0 --available_GPU 0 --use_info_batch
```

OrderDP:

```bash
python3 cifar/orderDP_example.py \
  --model r50 --dataset cifar100 --optimizer sgd --max-lr 0.03 \
  --random_len_ratio 0.8 --top_q_ratio 0.375 \
  --batch-size 128 --num_epoch 200 --available_GPU 0 --use_orderDP
```

The CIFAR scripts support both CIFAR-10 and CIFAR-100 through `--dataset`.

## ImageNet Experiments

Prepare ImageNet in the standard layout:

```text
IMAGENET_ROOT/
├── train/
└── val/
```

Then run from the repository root:

```bash
bash imagenet/r50_orderdp_90epoch.sh
```

or

```bash
bash imagenet/r50_unsup_90.sh
```

If you prefer to launch the scripts manually, the main entry points are:

- `imagenet/prune_experiment_orderdp.py`
- `imagenet/prune_experiment_unsup.py`

Update the ImageNet path in the shell scripts or pass your own dataset path on the command line.

## Notes

- The repository is self-contained and does not require an external InfoBatch checkout.
- Some utility implementations are adapted from the InfoBatch open-source release for fair comparison and reproducibility.
- Training logs, downloaded datasets, and checkpoints are excluded through `.gitignore`.

## Citation

If you find OrderDP useful in your research, please consider citing:

```bibtex
@inproceedings{
  jin2026orderdp,
  title={Order{DP}: A Theoretically Guaranteed Lossless Dynamic Data Pruning Framework},
  author={Chenhan Jin and Shengze Xu and Qingsong Wang and Fan JIA and Dingshuo Chen and Tieyong Zeng},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=e77QyyRQPz}
}
```

## Acknowledgements

This work builds on the open-source PyTorch ecosystem and prior research on data pruning and importance sampling. Our implementation is inspired in part by the InfoBatch codebase (`https://github.com/NUS-HPC-AI-Lab/InfoBatch`). We thank the InfoBatch authors and the broader community for releasing code that supported comparison and reproducibility.
