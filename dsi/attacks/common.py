"""Attack-runner abstractions: ASR computation, common loops over a callable.

The three concrete spaces (pixel/latent/embedding) each implement an `attack_step`
that maps (input, target) -> perturbed input. The runner here loops over prompts,
calls the step, evaluates the safety classifier, and aggregates ASR.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

from dsi.config import AttackSpace
from dsi.data.types import Prompt

Verdict = Literal["bypass", "blocked", "unknown"]


@dataclass
class AttackResult:
    prompt: Prompt
    verdict: Verdict = "unknown"
    pre_score: float = float("nan")
    post_score: float = float("nan")
    perturbation_norm: float = float("nan")
    ssim: float = float("nan")
    artifact_path: str = ""
    extra: dict = field(default_factory=dict)


@dataclass
class AttackBatchResult:
    space: AttackSpace
    eps: float
    results: list[AttackResult] = field(default_factory=list)

    @property
    def asr(self) -> float:
        n = len(self.results)
        if n == 0:
            return 0.0
        return sum(r.verdict == "bypass" for r in self.results) / n

    @property
    def mean_perturb(self) -> float:
        import statistics

        vals = [r.perturbation_norm for r in self.results if r.perturbation_norm == r.perturbation_norm]
        return statistics.mean(vals) if vals else float("nan")

    @property
    def mean_ssim(self) -> float:
        import statistics

        vals = [r.ssim for r in self.results if r.ssim == r.ssim]
        return statistics.mean(vals) if vals else float("nan")


def asr_from_verdicts(verdicts: list[Verdict]) -> float:
    n = len(verdicts)
    if n == 0:
        return 0.0
    return sum(v == "bypass" for v in verdicts) / n


def run_attack(
    prompts: list[Prompt],
    *,
    attack_step: Callable,
    eval_safety: Callable,
    space: AttackSpace,
    eps: float,
) -> AttackBatchResult:
    """Generic attack-runner.

    `attack_step(prompt) -> dict(perturbed, perturb_norm, ssim, artifact_path)`.
    `eval_safety(perturbed) -> dict(score, verdict)` where verdict ∈ {bypass, blocked}.
    """
    out = AttackBatchResult(space=space, eps=eps)
    for p in prompts:
        try:
            atk = attack_step(p)
        except NotImplementedError:
            raise
        except Exception as e:
            out.results.append(AttackResult(prompt=p, verdict="unknown", extra={"error": repr(e)}))
            continue
        ev = eval_safety(atk["perturbed"])
        out.results.append(
            AttackResult(
                prompt=p,
                verdict=ev["verdict"],
                pre_score=atk.get("pre_score", float("nan")),
                post_score=ev.get("score", float("nan")),
                perturbation_norm=atk.get("perturb_norm", float("nan")),
                ssim=atk.get("ssim", float("nan")),
                artifact_path=atk.get("artifact_path", ""),
            )
        )
    return out
