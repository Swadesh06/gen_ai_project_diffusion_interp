"""SAE hook plumbing for SDXL UNet.

Mirrors Surkov et al.'s `SDLens/hooked_sd_pipeline` interface in spirit:
  - register forward hooks on a set of UNet submodules (block names like "down.2.1")
  - capture pre-block input activations per call
  - optionally run them through an SAE encoder (collect features), and optionally
    swap the activation for `decode(intervened_z)` to perform a runtime patch

Use as a context manager so registration / removal is exception-safe and per-call.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterator

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore


HOOKPOINT_TO_GETTER: dict[str, Callable] = {
    # SDXL UNet block path lookup. The exact attribute path depends on diffusers' UNet;
    # populate at GPU-session time when we can introspect the live module.
    "down.2.1": lambda u: u.down_blocks[2].attentions[1],
    "mid.0": lambda u: u.mid_block.attentions[0],
    "up.0.0": lambda u: u.up_blocks[0].attentions[0],
    "up.0.1": lambda u: u.up_blocks[0].attentions[1],
}


@dataclass
class HookCapture:
    """Per-hookpoint per-step record."""

    hookpoint: str
    inputs: list = field(default_factory=list)   # raw block-input activations
    z: list = field(default_factory=list)        # SAE encoded features
    timesteps: list[int] = field(default_factory=list)


class SAEHookManager:
    """Manages forward-pre hooks on a set of UNet submodules.

    Use:
        sae_map = {"down.2.1": sae_d21, "mid.0": sae_mid, ...}
        mgr = SAEHookManager(unet, sae_map, capture=True)
        with mgr:
            pipe(prompt, ...)
        feats = mgr.captured                 # dict[str, HookCapture]
    """

    def __init__(
        self,
        unet,
        saes: dict[str, "SparseAutoencoder"],  # noqa: F821
        *,
        capture: bool = True,
        intervene_fn: Callable | None = None,
        device: str = "cpu",
    ):
        if torch is None:
            raise ImportError("PyTorch required for SAEHookManager")
        self.unet = unet
        self.saes = saes
        self.capture = capture
        self.intervene_fn = intervene_fn
        self.device = device
        self.captured: dict[str, HookCapture] = {hp: HookCapture(hp) for hp in saes}
        self._handles: list = []
        self._step_counter: int = 0

    def reset(self) -> None:
        self.captured = {hp: HookCapture(hp) for hp in self.saes}
        self._step_counter = 0

    def step(self) -> None:
        self._step_counter += 1

    def _make_hook(self, hookpoint: str):
        sae = self.saes[hookpoint]

        def pre_hook(module, args, kwargs):
            x = args[0] if args else kwargs.get("hidden_states")
            if x is None:
                return None
            z = None
            with torch.no_grad():
                z = sae.encode(x)
            if self.capture:
                cap = self.captured[hookpoint]
                cap.inputs.append(x.detach().cpu())
                cap.z.append(z.detach().cpu())
                cap.timesteps.append(self._step_counter)
            if self.intervene_fn is not None:
                z_new = self.intervene_fn(hookpoint, z, self._step_counter)
                if z_new is not None:
                    x_new = sae.decode(z_new)
                    new_args = (x_new,) + args[1:]
                    return (new_args, kwargs)
            return None

        return pre_hook

    def __enter__(self):
        for hp in self.saes:
            try:
                target = HOOKPOINT_TO_GETTER[hp](self.unet)
            except (AttributeError, IndexError, KeyError):
                continue
            h = target.register_forward_pre_hook(self._make_hook(hp), with_kwargs=True)
            self._handles.append(h)
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self._handles:
            h.remove()
        self._handles.clear()


@contextmanager
def sae_capture(unet, saes: dict, **kw) -> Iterator[SAEHookManager]:
    mgr = SAEHookManager(unet, saes, **kw)
    with mgr:
        yield mgr
