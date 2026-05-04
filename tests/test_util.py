"""Util tests: seed, ckpt, logging."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from dsi.util.ckpt import RollingCheckpointer
from dsi.util.logging import get_logger
from dsi.util.seed import capture_rng, restore_rng, set_seed


def test_set_seed_reproducible():
    set_seed(42)
    a = (random.random(), float(np.random.rand()), float(torch.rand(()).item()))
    set_seed(42)
    b = (random.random(), float(np.random.rand()), float(torch.rand(()).item()))
    assert a == b


def test_capture_restore_rng():
    set_seed(0)
    s = capture_rng()
    a = float(torch.rand(()).item())
    restore_rng(s)
    b = float(torch.rand(()).item())
    assert a == b


def test_logger_writes_to_file(tmp_path: Path):
    log_path = tmp_path / "x.log"
    log = get_logger("dsi.test", log_path=log_path)
    log.info("hello")
    contents = log_path.read_text()
    assert "hello" in contents


def test_rolling_checkpointer(tmp_path: Path):
    ckptr = RollingCheckpointer(tmp_path / "ck", keep_n=2)
    head = torch.nn.Linear(4, 1)
    opt = torch.optim.SGD(head.parameters(), lr=0.01)
    p = ckptr.save(step=1, epoch=0, model_state=head.state_dict(),
                   optimizer_state=opt.state_dict(), scheduler_state=None)
    assert p.exists()
    p2 = ckptr.save(step=2, epoch=0, model_state=head.state_dict(),
                    optimizer_state=opt.state_dict(), scheduler_state=None)
    p3 = ckptr.save(step=3, epoch=0, model_state=head.state_dict(),
                    optimizer_state=opt.state_dict(), scheduler_state=None)
    files = sorted((tmp_path / "ck").glob("step_*.pt"))
    assert len(files) == 2  # rolled
    last = (tmp_path / "ck" / "last.pt")
    assert last.exists()
    payload = ckptr.load(last)
    assert "model_state_dict" in payload


def test_ckpt_best_metric(tmp_path: Path):
    ckptr = RollingCheckpointer(tmp_path / "ck2", keep_n=4)
    head = torch.nn.Linear(2, 1)
    ckptr.save(1, 0, head.state_dict(), metric=0.5, better="lower")
    ckptr.save(2, 0, head.state_dict(), metric=0.3, better="lower")
    ckptr.save(3, 0, head.state_dict(), metric=0.7, better="lower")
    assert (tmp_path / "ck2" / "best.pt").exists()
    assert ckptr.best_metric == 0.3
