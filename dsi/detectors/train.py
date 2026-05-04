"""Detector training loop with checkpointing/resume per CLAUDE.md §7.

GPU is needed for the forward pass; everything around it (DataLoader, optimizer,
scheduler, ckpt rollover, WandB) is CPU-runnable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from dsi.config import DetectorCfg, cfg
from dsi.util.ckpt import RollingCheckpointer
from dsi.util.logging import get_logger
from dsi.util.seed import restore_rng, set_seed
from dsi.util.wandb import init_wandb


@dataclass
class TrainState:
    step: int = 0
    epoch: int = 0
    best_metric: float = float("inf")
    history: list = field(default_factory=list)


@dataclass
class DetectorTrainCfg:
    exp_id: str = "B00_smoke"
    detector: DetectorCfg = field(default_factory=DetectorCfg)
    seed: int = 0
    device: str = "cpu"


def train_detector(
    *,
    cfg_train: DetectorTrainCfg,
    head_factory: Callable,
    train_loader,
    val_loader,
    loss_fn: Callable | None = None,
    metric_fn: Callable | None = None,
    resume: str | None = None,
    tags: list[str] | None = None,
):
    """Generic detector training loop.

    `head_factory(in_dim) -> nn.Module`. `loss_fn(logits, y) -> scalar`.
    `metric_fn(logits, y) -> dict` (e.g., {"auc": ..., "ap": ...}).
    """
    import torch
    import torch.nn.functional as F

    log = get_logger(cfg_train.exp_id, log_path=cfg.paths.log_root / f"{cfg_train.exp_id}.log")
    set_seed(cfg_train.seed)
    state = TrainState()

    sample = next(iter(train_loader))
    x_sample = sample[0] if isinstance(sample, (list, tuple)) else sample["x"]
    in_dim = x_sample.shape[-1] if x_sample.ndim >= 2 else x_sample.numel()
    head = head_factory(in_dim).to(cfg_train.device)

    opt = torch.optim.AdamW(head.parameters(), lr=cfg_train.detector.lr,
                            weight_decay=cfg_train.detector.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg_train.detector.epochs)

    ckptr = RollingCheckpointer(
        cfg.paths.checkpoint_root / cfg_train.exp_id, keep_n=cfg_train.detector.keep_last_n_ckpts,
    )

    if resume == "latest":
        last = ckptr.dir / "last.pt"
        if last.exists():
            ck = ckptr.load(last)
            head.load_state_dict(ck["model_state_dict"])
            opt.load_state_dict(ck["optimizer_state_dict"])
            sched.load_state_dict(ck["scheduler_state_dict"])
            restore_rng(ck["rng_state"])
            state.step = ck["step"]
            state.epoch = ck["epoch"]
            log.info(f"resumed from step={state.step} epoch={state.epoch}")
    elif resume:
        ck = ckptr.load(resume)
        head.load_state_dict(ck["model_state_dict"])

    run = init_wandb(cfg_train.exp_id, asdict(cfg_train), tags=(tags or []) + ["detector"])

    loss_fn = loss_fn or (lambda logits, y: F.binary_cross_entropy_with_logits(logits, y.float()))

    for epoch in range(state.epoch, cfg_train.detector.epochs):
        head.train()
        for batch in train_loader:
            x, y = (batch[0], batch[1]) if isinstance(batch, (list, tuple)) else (batch["x"], batch["y"])
            x = x.to(cfg_train.device)
            y = y.to(cfg_train.device)
            logits = head(x)
            loss = loss_fn(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            state.step += 1
            if run is not None:
                run.log({"train/loss": float(loss.detach()), "lr": opt.param_groups[0]["lr"],
                         "epoch": epoch, "step": state.step})
            if state.step % cfg_train.detector.ckpt_every_steps == 0:
                ckptr.save(state.step, epoch, head.state_dict(), opt.state_dict(),
                           sched.state_dict(), config=cfg_train, metric=float(loss.detach()))
        sched.step()

        head.eval()
        if metric_fn is not None:
            with torch.no_grad():
                logits_all, y_all = [], []
                for batch in val_loader:
                    x, y = (batch[0], batch[1]) if isinstance(batch, (list, tuple)) else (batch["x"], batch["y"])
                    logits_all.append(head(x.to(cfg_train.device)).cpu())
                    y_all.append(y)
                import torch as T

                metrics = metric_fn(T.cat(logits_all), T.cat(y_all))
                state.history.append({"epoch": epoch, **metrics})
                if run is not None:
                    run.log({f"val/{k}": v for k, v in metrics.items()} | {"epoch": epoch})
                primary = metrics.get("loss") or -metrics.get("auc", 0.0)
                ckptr.save(state.step, epoch + 1, head.state_dict(), opt.state_dict(),
                           sched.state_dict(), config=cfg_train, metric=primary)
                if primary < state.best_metric:
                    state.best_metric = primary

    if run is not None:
        run.finish()

    history_path = cfg.paths.log_root / f"{cfg_train.exp_id}_history.json"
    history_path.write_text(json.dumps(state.history, indent=2))
    return head, state


def smoke_train(in_dim: int = 16, n: int = 64, device: str = "cpu") -> dict:
    """5-step CPU smoke test on synthetic data. No WandB. Returns final loss."""
    import torch
    import torch.nn.functional as F

    set_seed(0)
    x = torch.randn(n, in_dim)
    w = torch.randn(in_dim)
    y = (x @ w > 0).float()
    head = torch.nn.Linear(in_dim, 1)
    opt = torch.optim.SGD(head.parameters(), lr=0.1)
    losses = []
    for _ in range(5):
        logits = head(x).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return {"losses": losses}
