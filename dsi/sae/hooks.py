"""SAE hook plumbing for SDXL UNet.

Two hook managers:

  - SurkovHookManager: Surkov-style residual hooks. Hook the output of
    `Transformer2DModel` blocks at `down.2.1`, `mid.0`, `up.0.0`, `up.0.1`,
    compute `diff = output - input` (residual contribution), permute (B, C, H, W)
    → (B, H, W, C), normalize by per-block (mean, std), encode with the TopK SAE.
    This matches the protocol in sdxl-unbox/utils/hooks.py and SDLens.

  - GenericHookManager: pre-hook on a sequence-style block (B, T, C). Used for
    SAeUron-style hookpoints where the input is already a flat (B, T, C) sequence.

Both expose the same `with mgr: pipe(prompt, ...)` context-manager API and a
`captured` dict[str, HookCapture] of per-step per-block records.
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
    # SDXL UNet block path lookup. The exact attribute path is consistent with
    # surkovv/sdxl-unbox's `code_to_block` map.
    "down.2.1": lambda u: u.down_blocks[2].attentions[1],
    "mid.0":    lambda u: u.mid_block.attentions[0],
    "up.0.0":   lambda u: u.up_blocks[0].attentions[0],
    "up.0.1":   lambda u: u.up_blocks[0].attentions[1],
    # SAeUron's SD v1.5 hookpoints
    "up.1.1":   lambda u: u.up_blocks[1].attentions[1],
    "up.1.2":   lambda u: u.up_blocks[1].attentions[2],
}


@dataclass
class HookCapture:
    """Per-hookpoint per-step record."""

    hookpoint: str
    inputs: list = field(default_factory=list)   # raw block-input activations
    z: list = field(default_factory=list)        # SAE encoded features
    timesteps: list[int] = field(default_factory=list)


def _retrieve(io):
    """Extract the primary tensor from a tuple / dataclass / tensor return."""
    if io is None:
        return None
    if isinstance(io, tuple):
        return io[0]
    if hasattr(io, "sample"):  # diffusers Transformer2DModelOutput
        return io.sample
    return io


def _replace_primary(orig, new_tensor):
    """Mirror image of `_retrieve`: put `new_tensor` back in the same container as `orig`."""
    if isinstance(orig, tuple):
        return (new_tensor,) + orig[1:]
    if hasattr(orig, "sample"):
        orig.sample = new_tensor
        return orig
    return new_tensor


class SurkovHookManager:
    """Surkov residual hooks on `Transformer2DModel` blocks (post-forward).

    Use:
        saes = {hp: load_surkov_sae(hp) for hp in ("down.2.1","mid.0","up.0.0","up.0.1")}
        norms = {hp: load_surkov_norm(hp) for hp in saes}
        mgr = SurkovHookManager(unet, saes, norms=norms, capture=True)
        with mgr:
            pipe(prompt, ...)
        feats = mgr.captured                 # dict[hookpoint -> HookCapture]
    """

    def __init__(
        self,
        unet,
        saes: dict,
        *,
        norms: dict | None = None,
        capture: bool = True,
        intervene_fn: Callable | None = None,
        device: str | None = None,
        keep_inputs: bool = False,
        attack_mode: bool = False,
    ):
        if torch is None:
            raise ImportError("PyTorch required")
        self.unet = unet
        self.saes = saes
        self.norms = norms or {}
        self.capture = capture
        self.intervene_fn = intervene_fn
        self.device = device
        self.keep_inputs = keep_inputs
        # attack_mode: keep z gradient-attached and on-device for attack-time
        # backprop (e.g., PGD that needs gradient through SAE encode). When
        # True, also stores the live z tensor (not detached cpu) so callers
        # can read it via mgr.captured[hp].z[-1] for loss computation.
        self.attack_mode = attack_mode
        self.captured: dict[str, HookCapture] = {hp: HookCapture(hp) for hp in saes}
        self._handles: list = []
        self._step_counter: int = 0

    def reset(self) -> None:
        self.captured = {hp: HookCapture(hp) for hp in self.saes}
        self._step_counter = 0

    def step(self) -> None:
        self._step_counter += 1

    def _normalize(self, x_bhwc, hookpoint: str):
        mu, std = self.norms.get(hookpoint, (None, None))
        if mu is None or std is None:
            return x_bhwc
        return (x_bhwc - mu.to(x_bhwc.device).to(x_bhwc.dtype)) / std.to(x_bhwc.device).to(x_bhwc.dtype).clamp(min=1e-8)

    def _denormalize(self, x_bhwc, hookpoint: str):
        mu, std = self.norms.get(hookpoint, (None, None))
        if mu is None or std is None:
            return x_bhwc
        return x_bhwc * std.to(x_bhwc.device).to(x_bhwc.dtype) + mu.to(x_bhwc.device).to(x_bhwc.dtype)

    def _make_hook(self, hookpoint: str):
        sae = self.saes[hookpoint]

        def post_hook(module, args, kwargs, output):
            inp = _retrieve(args if args else kwargs.get("hidden_states", None))
            out = _retrieve(output)
            if inp is None or out is None:
                return None
            diff = out - inp                        # residual contribution
            x_bhwc = diff.permute(0, 2, 3, 1)       # (B, H, W, C)
            x_norm = self._normalize(x_bhwc, hookpoint).to(next(sae.parameters()).device)
            if self.attack_mode:
                # Keep the encode in the autograd graph so attack-time PGD can
                # backprop loss through z to the input image.
                z = sae.encode(x_norm.to(next(sae.parameters()).dtype))
            else:
                with torch.no_grad():
                    z = sae.encode(x_norm.to(next(sae.parameters()).dtype))
            if self.capture:
                cap = self.captured[hookpoint]
                if self.attack_mode:
                    # Live z tensor on device, gradient attached. Caller reads
                    # cap.z[-1] for loss computation.
                    if self.keep_inputs:
                        cap.inputs.append(diff)
                    cap.z.append(z)
                else:
                    if self.keep_inputs:
                        cap.inputs.append(diff.detach().cpu())
                    cap.z.append(z.detach().cpu())
                cap.timesteps.append(self._step_counter)
            if self.intervene_fn is not None:
                z_new = self.intervene_fn(hookpoint, z, self._step_counter)
                if z_new is not None:
                    rec_norm = sae.decode(z_new)
                    rec = self._denormalize(rec_norm, hookpoint)
                    rec_bchw = rec.permute(0, 3, 1, 2).to(out.dtype).to(out.device)
                    return _replace_primary(output, inp + rec_bchw)
            return None

        return post_hook

    def __enter__(self):
        for hp in self.saes:
            try:
                target = HOOKPOINT_TO_GETTER[hp](self.unet)
            except (AttributeError, IndexError, KeyError):
                continue
            h = target.register_forward_hook(self._make_hook(hp), with_kwargs=True)
            self._handles.append(h)
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self._handles:
            h.remove()
        self._handles.clear()


class GenericPreHookManager:
    """Pre-hook variant for sequence-style blocks (B, T, C) — kept for SAeUron and tests.

    Same context-manager API as SurkovHookManager but registers a forward-pre-hook
    on the block input rather than the residual.
    """

    def __init__(self, unet, saes: dict, *, capture: bool = True, intervene_fn=None,
                 device: str | None = None):
        if torch is None:
            raise ImportError("PyTorch required")
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


# Back-compat alias used in older smoke scripts and tests.
SAEHookManager = SurkovHookManager


@contextmanager
def sae_capture(unet, saes: dict, **kw) -> Iterator[SurkovHookManager]:
    mgr = SurkovHookManager(unet, saes, **kw)
    with mgr:
        yield mgr
