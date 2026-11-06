"""Train the original model, and the retrain-on-retain reference.

SGD momentum 0.9, lr 0.1, cosine schedule, weight decay 5e-4, batch 128, 200 epochs,
random crop(32, pad 4) + horizontal flip. Checkpoints every epoch; supports resume.
"""

import os
import time

import torch
import torch.nn as nn

from ..eval.accuracy import evaluate_all
from ..models.resnet_cifar import resnet18_cifar


def build_optimizer(model, lr=0.1, momentum=0.9, weight_decay=5e-4, epochs=200):
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                          weight_decay=weight_decay, nesterov=False)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    return opt, sched


def save_ckpt(path, model, opt, sched, epoch, config, extra=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    torch.save({
        "model": model.state_dict(),
        "optimizer": opt.state_dict() if opt is not None else None,
        "scheduler": sched.state_dict() if sched is not None else None,
        "epoch": epoch,
        "config": config,
        "extra": extra or {},
    }, tmp)
    os.replace(tmp, path)  # atomic: a kill mid-save cannot corrupt the checkpoint


def load_ckpt(path, model, opt=None, sched=None, map_location="cpu"):
    ck = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ck["model"])
    if opt is not None and ck.get("optimizer"):
        opt.load_state_dict(ck["optimizer"])
    if sched is not None and ck.get("scheduler"):
        sched.load_state_dict(ck["scheduler"])
    return ck


def train_model(loader, loaders, device, epochs=200, lr=0.1, weight_decay=5e-4,
                momentum=0.9, ckpt_path=None, resume=False, config=None,
                log_every=1, amp=True, label="base",
                forget_class=0, log_fh=None, num_classes=10, model=None):
    """Standard supervised training. `loader` is the augmented loader to train on.

    Returns (model, history). `loaders` is the dict from data.cifar.build_loaders,
    used only for periodic evaluation. `model` defaults to the CIFAR ResNet-18.
    """
    model = (model or resnet18_cifar(num_classes=num_classes, norm="bn")).to(device)
    opt, sched = build_optimizer(model, lr=lr, momentum=momentum,
                                 weight_decay=weight_decay, epochs=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    crit = nn.CrossEntropyLoss()

    start_epoch = 0
    history = []
    if resume and ckpt_path and os.path.exists(ckpt_path):
        ck = load_ckpt(ckpt_path, model, opt, sched, map_location=device)
        start_epoch = int(ck.get("epoch", 0))
        history = ck.get("extra", {}).get("history", [])
        _log(log_fh, f"[{label}] resumed from {ckpt_path} at epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        model.train()
        t0 = time.time()
        tot_loss, tot_correct, tot_n = 0.0, 0, 0
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                out = model(x)
                loss = crit(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot_loss += loss.item() * y.size(0)
            tot_correct += (out.argmax(1) == y).sum().item()
            tot_n += y.size(0)
        sched.step()

        rec = {
            "epoch": epoch + 1,
            "train_loss": tot_loss / max(tot_n, 1),
            "train_acc": 100.0 * tot_correct / max(tot_n, 1),
            "lr": sched.get_last_lr()[0],
            "secs": round(time.time() - t0, 1),
        }
        if (epoch + 1) % log_every == 0 or epoch + 1 == epochs:
            m = evaluate_all(model, loaders, device)
            rec.update({f"eval_{k}": v for k, v in m.items()})
            _log(log_fh,
                 f"[{label}] ep {epoch+1:3d}/{epochs} loss {rec['train_loss']:.4f} "
                 f"tracc {rec['train_acc']:.2f} | test {m['test_full']:.2f} "
                 f"| F(train) {m['forget_train']:.2f} R(train) {m['retain_train']:.2f} "
                 f"| {rec['secs']}s")
        history.append(rec)

        if ckpt_path:
            save_ckpt(ckpt_path, model, opt, sched, epoch + 1, config or {},
                      extra={"history": history})

    return model, history


def _log(fh, msg):
    print(msg, flush=True)
    if fh is not None:
        fh.write(msg + "\n")
        fh.flush()
