from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy.interpolate import CubicSpline


@dataclass(frozen=True)
class Subtitle:
    start: float
    end: float
    text: str = ""


@dataclass
class VSIConfig:
    top_k: int = 8
    samples_per_round: int = 16
    detection_budget: int = 64
    text_weight: float = 0.3
    object_threshold: float = 0.7
    text_threshold: float = 0.2
    soft_threshold: float = 0.5
    soft_amplification: float = 2.0
    subtitle_extension_seconds: float = 2.0
    probability_floor: float | None = None
    seed: int = 0

    def validate(self, n_frames: int) -> None:
        if n_frames < 1:
            raise ValueError("n_frames must be positive")
        if self.top_k < 1 or self.samples_per_round < 1 or self.detection_budget < 1:
            raise ValueError("frame counts and budgets must be positive")
        if not 0 <= self.text_weight <= 1:
            raise ValueError("text_weight must be in [0, 1]")


@dataclass
class VSIResult:
    frame_indices: list[int]
    timestamps: list[float]
    fused_scores: np.ndarray
    sampling_probabilities: np.ndarray
    visited_indices: list[int]
    rounds: int
    history: list[np.ndarray] = field(default_factory=list)


def soft_threshold(scores: Sequence[float], theta: float = 0.5, gamma: float = 2.0) -> np.ndarray:
    """Equation (8): B = min(M + gamma * max(M-theta, 0), 1)."""
    x = np.asarray(scores, dtype=float)
    return np.minimum(x + gamma * np.maximum(x - theta, 0.0), 1.0)


def subtitle_frame_scores(
    subtitles: Sequence[Subtitle], similarities: Sequence[float], n_frames: int, fps: float,
    *, threshold: float = 0.2, extension: float = 2.0,
) -> np.ndarray:
    """Equations (9)-(10): Gaussian propagation and max aggregation."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    if len(subtitles) != len(similarities):
        raise ValueError("subtitles and similarities must have equal length")
    times = np.arange(n_frames, dtype=float) / fps
    output = np.zeros(n_frames, dtype=float)
    for sub, score in zip(subtitles, similarities):
        if score <= threshold or sub.end < sub.start:
            continue
        center = (sub.start + sub.end) / 2.0
        sigma = (sub.end - sub.start + 2.0 * extension) / 4.0
        mask = (times >= sub.start - extension) & (times <= sub.end + extension)
        if sigma > 0 and np.any(mask):
            propagated = score * np.exp(-((times[mask] - center) ** 2) / (2.0 * sigma**2))
            output[mask] = np.maximum(output[mask], propagated)
    return output


def zscore(values: Sequence[float]) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    std = float(np.std(x))
    return (x - float(np.mean(x))) / (std + 1e-6)


def fuse_scores(object_scores: Sequence[float], text_scores: Sequence[float], text_weight: float) -> np.ndarray:
    """Equation (11), retaining zero contribution for an absent/constant branch."""
    obj, text = np.asarray(object_scores, float), np.asarray(text_scores, float)
    if obj.shape != text.shape:
        raise ValueError("object_scores and text_scores must have equal shape")
    norm_obj = np.zeros_like(obj) if np.ptp(obj) == 0 else zscore(obj)
    norm_text = np.zeros_like(text) if np.ptp(text) == 0 else zscore(text)
    return text_weight * norm_text + (1.0 - text_weight) * norm_obj


def interpolate_distribution(
    visited: Sequence[int], scores: Sequence[float], n_frames: int, floor: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Equations (13)-(15): spline, lower bound, then normalized sigmoid."""
    x = np.asarray(sorted(set(visited)), dtype=int)
    if x.size == 0:
        interpolated = np.zeros(n_frames, dtype=float)
    elif x.size == 1:
        interpolated = np.full(n_frames, float(scores[x[0]]))
    else:
        # Linear fallback for two points; natural cubic avoids unstable edge extrapolation.
        grid = np.arange(n_frames)
        if x.size == 2:
            interpolated = np.interp(grid, x, np.asarray(scores)[x])
        else:
            interpolated = CubicSpline(x, np.asarray(scores)[x], bc_type="natural", extrapolate=True)(grid)
    corrected = np.maximum(1.0 / n_frames if floor is None else floor, interpolated)
    shifted = corrected - np.max(corrected)
    sigmoid = 1.0 / (1.0 + np.exp(-np.clip(shifted, -60, 60)))
    return corrected, sigmoid / sigmoid.sum()


def _uniform_indices(n_frames: int, count: int) -> np.ndarray:
    return np.unique(np.rint(np.linspace(0, n_frames - 1, min(count, n_frames))).astype(int))


def select_keyframes(
    n_frames: int,
    fps: float,
    object_scorer: Callable[[Sequence[int]], Sequence[float]],
    *,
    text_scores: Sequence[float] | None = None,
    config: VSIConfig | None = None,
    stop_when_found: Callable[[Sequence[int]], bool] | None = None,
) -> VSIResult:
    """Run Algorithm 1 with a pluggable batched object detector.

    ``object_scorer(indices)`` returns one [0,1] score per requested frame,
    corresponding to Equation (4)'s max(confidence * object importance).
    """
    cfg = config or VSIConfig()
    cfg.validate(n_frames)
    text = np.zeros(n_frames) if text_scores is None else np.asarray(text_scores, float)
    if text.shape != (n_frames,):
        raise ValueError("text_scores must have shape (n_frames,)")
    rng = np.random.default_rng(cfg.seed)
    object_scores = np.zeros(n_frames, dtype=float)
    visited: set[int] = set()
    probabilities = np.full(n_frames, 1.0 / n_frames)
    history: list[np.ndarray] = []
    rounds = 0

    while len(visited) < min(cfg.detection_budget, n_frames):
        remaining_budget = min(cfg.samples_per_round, cfg.detection_budget - len(visited), n_frames - len(visited))
        available = np.array(sorted(set(range(n_frames)) - visited), dtype=int)
        if rounds == 0:
            chosen = _uniform_indices(n_frames, remaining_budget)
            chosen = np.array([i for i in chosen if i not in visited], dtype=int)
        else:
            p = probabilities[available]
            p = p / p.sum()
            chosen = rng.choice(available, size=remaining_budget, replace=False, p=p)
        if chosen.size == 0:
            break
        batch_scores = np.asarray(object_scorer(chosen.tolist()), dtype=float)
        if batch_scores.shape != chosen.shape:
            raise ValueError("object_scorer must return exactly one score per frame")
        object_scores[chosen] = np.clip(batch_scores, 0.0, 1.0)
        visited.update(chosen.tolist())
        observed_fused = fuse_scores(object_scores, text, cfg.text_weight)
        _, probabilities = interpolate_distribution(sorted(visited), observed_fused, n_frames, cfg.probability_floor)
        history.append(probabilities.copy())
        rounds += 1
        if stop_when_found is not None and stop_when_found(sorted(visited)):
            break

    final = fuse_scores(object_scores, text, cfg.text_weight)
    # Only detector-visited frames are valid visual results. This prevents an
    # unvisited text peak being returned without visual confirmation.
    candidates = np.asarray(sorted(visited), dtype=int)
    order = candidates[np.argsort(-final[candidates], kind="stable")]
    selected = sorted(order[: min(cfg.top_k, len(order))].tolist())
    return VSIResult(selected, [i / fps for i in selected], final, probabilities,
                     sorted(visited), rounds, history)
