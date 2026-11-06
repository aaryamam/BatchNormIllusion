# The BatchNorm Illusion

Code for *The BatchNorm Illusion: Diagnosing Normalization Artifacts in Machine Unlearning Evaluation*.

A single forward pass over retain data recomputes a model's BatchNorm running statistics without changing any weight. This repository provides that diagnostic, the unlearning methods we evaluate with it, and the measurements around them.

## Using the diagnostic

```python
from src.diagnostic.recalibrate import recalibrate_bn
from src.eval.accuracy import accuracy

pre_f = accuracy(model, forget_loader, device)
model, info = recalibrate_bn(model, retain_loader, device)
post_f = accuracy(model, forget_loader, device)

delta_f = post_f - pre_f
assert info["weights_bit_identical"]
```

`recalibrate_bn` wraps `torch.optim.swa_utils.update_bn` and requires at least 5,000 retain samples by default (`strict=False` disables the check).

## Setup

```bash
pip install -r requirements.txt
```

Use Python 3.10. CIFAR-10 and CIFAR-100 download to `./data`; set `DATA_ROOT` to use another location. Run the scripts from the repository root. Checkpoints go to `checkpoints/` and results to `results/`.

Tiny-ImageNet is read from `<DATA_ROOT>/tiny-imagenet-200`, unpacked from [the original archive](https://cs231n.stanford.edu/tiny-imagenet-200.zip). ImageNet-100 ([`clane9/imagenet-100`](https://huggingface.co/datasets/clane9/imagenet-100)) and the pretrained weights download on first use.

## Running the experiments

Train the base model and the retrain references:

```bash
python scripts/train_base.py --which base
```

```bash
for s in 0 1 2; do python scripts/train_base.py --which retrain --seed $s; done
```

Run the unlearning methods and measure the recalibration gap, then evaluate the saved checkpoints:

```bash
python scripts/run_unlearning.py --seeds 0 1 2
```

```bash
python scripts/run_evaluation.py
```

```bash
python scripts/run_recovery.py
```

The remaining experiments need only the base model, except that `run_imagenet100.py` needs no trained model and `run_cifar100.py` needs the CIFAR-100 base and retrain models from `train_base.py --dataset cifar100`, including a five-class retrain (`--forget-classes 0 1 2 3 4`).

```bash
python scripts/run_sweeps.py
```

```bash
python scripts/run_mia.py
```

```bash
python scripts/run_cifar100.py
```

```bash
python scripts/run_instance_wise.py
```

```bash
python scripts/run_batch_composition.py
```

```bash
python scripts/run_random_labeling.py
```

```bash
python scripts/run_bn_analysis.py
```

```bash
python scripts/run_divergence.py
```

```bash
python scripts/run_imagenet100.py
```

GPM-W is run separately and passed to the recovery script:

```bash
python scripts/run_unlearning.py --methods gpm_w --out gpm_w.json
```

```bash
python scripts/run_recovery.py --input gpm_w.json --budgets 100 --out recovery_gpm_w.json
```

### Other architectures

The GroupNorm control, Tiny-ImageNet and ViT experiments need their own models. `run_tiny_imagenet.py` needs one retrain per seed.

```bash
for w in base retrain; do python scripts/train_base.py --which $w --arch resnet18_gn; done
```

```bash
python scripts/run_groupnorm.py
```

```bash
for w in base retrain; do python scripts/train_base.py --which $w --dataset tiny_imagenet --arch resnet50 --eval-every 10; done
```

```bash
for s in 1 2; do python scripts/train_base.py --which retrain --dataset tiny_imagenet --arch resnet50 --eval-every 10 --seed $s; done
```

```bash
python scripts/run_tiny_imagenet.py
```

```bash
for w in base retrain; do python scripts/train_base.py --which $w --dataset tiny_imagenet --arch vit_s16 --epochs 10 --lr 0.01 --weight-decay 0; done
```

```bash
python scripts/run_vit.py --override ga.lr=1e-4 ga.steps=200
```

`run_groupnorm.py`, `run_tiny_imagenet.py` and `run_vit.py` take per-method overrides as `METHOD.KEY=VALUE`.

| Script | Measures |
|---|---|
| `run_unlearning.py` | Forget and retain accuracy before and after recalibration, ΔF |
| `run_evaluation.py` | LP-50, NCC, Relearn-AUC, the BN/NC decomposition, cross-correlation, rank correlations |
| `run_recovery.py` | Recovery across attacker budgets, in-distribution and out-of-distribution |
| `run_sweeps.py` | GA step count; learning-rate and step grids for BadTeacher and SalUn |
| `run_mia.py` | Membership-inference AUC before and after recalibration |
| `run_cifar100.py` | CIFAR-100 single-class and five-class forgetting |
| `run_instance_wise.py` | ΔF when forget samples are drawn across all classes |
| `run_batch_composition.py` | ΔF across retain fractions of the unlearning loader |
| `run_random_labeling.py` | RandomLabeling with BN in train and eval mode |
| `run_bn_analysis.py` | Layerwise BN mismatch (with plot), batch-size sensitivity, idempotence |
| `run_divergence.py` | Forget/retain activation divergence at every BN layer |
| `run_groupnorm.py` | GroupNorm control on ResNet-18 |
| `run_tiny_imagenet.py` | ResNet-50 on Tiny-ImageNet: method panel and GA step sweep |
| `run_vit.py` | LayerNorm control on ViT-S/16 |
| `run_imagenet100.py` | BadTeacher on a pretrained ResNet-50, with ten-image recovery |

## Methods

| Name | Method | BN mode |
|---|---|---|
| `scrub` | SCRUB | train |
| `ga` | Gradient ascent | train |
| `ga_ft` | Gradient ascent + retain fine-tune | train |
| `salun` | SalUn | train |
| `badteacher` | Bad Teaching | eval |
| `incompteacher` | Competent/incompetent teacher | train |
| `noise_inject` | Gaussian noise + retain repair | train |
| `ssd` | Selective Synaptic Dampening | train |
| `random_labeling` | Random relabelling | train |
| `gpm_w` | Weight-projection gradient ascent | train |

Defaults are in each module's `DEFAULTS`. Override them from the command line:

```bash
python scripts/run_unlearning.py --methods salun --override steps=100 lr=0.002
```

The `noise_inject` default of σ = 5 can destroy a converged network, in which case lower it with `--override sigma_noise=0.5`.

Recalibration uses the augmented retain stream (`retain_recal_aug`), matching the distribution seen during training. `run_mia.py`, `run_instance_wise.py`, `run_batch_composition.py` and `run_random_labeling.py` use `retain_recal`, the deterministic alternative, which is exactly repeatable because its batches are fixed. `run_bn_analysis.py` uses a fixed 5,000-sample retain subset.

## Citation

```bibtex
@article{kalani2026batchnorm,
  title   = {The BatchNorm Illusion: Diagnosing Normalization Artifacts in Machine Unlearning Evaluation},
  author  = {Kalani, Aaryaman and Mandal, Murari and Kumar, Dhruv and Kankanhalli, Mohan and Sinha, Yash},
  year    = {2026}
}
```
