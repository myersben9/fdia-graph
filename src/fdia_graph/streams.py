"""Continuous attacked streams — the realistic time series for temporal models (LSTM / TGN).

The classification shard (`generate`) is a SHUFFLED table of independent labeled snapshots: attacks are
injected at scattered timesteps and mixed together, so its rows are not contiguous in time. A temporal model
wants the opposite: one running timeline where the grid operates normally and an attack appears over a
contiguous EPISODE, then clears. `generate_stream` produces exactly that, per system:

    Three aligned measurement layers per frame ([|V|, Pinj, Qinj, angle] columns):
    node_x   [T, N, 4]  OBSERVED feed — attacked+noisy where attacked, benign+noisy elsewhere (the model input)
    benign   [T, N, 4]  the same meters with the ATTACK REMOVED (benign+noisy) — what they would read un-attacked
    clean    [T, N, 4]  NOISELESS attack-free TRUE state — the SE / reconstruction target
                        (benign-clean = noise always; node_x-benign = attack exactly for Ad/As/Ar only.
                        Aq/At/Al re-solve, so node_x-benign there also carries a noise term; use clean as SE target.)
    edge_x, edge_benign, edge_clean [T, E, 2]  the SAME three layers for branch flows [P_from, Q_from]
                        (observed / attack-removed / noiseless). Node + edge together are the full SE measurement set.
    edge_index [2, E]    static graph connectivity (COO, int64 -> torch.long); edge_attr [E, 8] static line
                        features (r, x, b, g, gs, bs, tap, shift) — the two holders PyTorch-Geometric models expect.
    node_m [N, 4], edge_m [E, 2]  static meter-availability masks (metering is SPARSE; 0 = no meter, entry is
                        zero-filled). The benign-clean = meter-noise identity holds on measured channels for all
                        families; the node_x - benign = attack identity holds exactly only for Ad/As/Ar (see above).
    y        [T, N]      per-timestep, per-bus attack label (0 on benign frames/buses)
    family   [T]         active attack family id at each timestep (0 = benign)
    temporal_delta, swing [T, N, 2]   change vs the PREVIOUS EMITTED frame (see note below)
    episodes list        (onset, length, family, attacked buses) for every attack episode

Temporal-feature note: unlike the shard (which compares an attacked snapshot to the benign X[t-1]), a stream
compares each frame to the previous EMITTED frame. That is what keeps a stealthy ramp looking like a small
per-step change and a spike looking like an abrupt jump — the spike-vs-ramp signal the dataset is built on.
"""

from __future__ import annotations

import numbers
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np

from .engine import FAM_ID, FdiaGenerator
from .engine.records import RAMP_FAMILY, SINGLE_SHOT_ORDER, FrameKnobs
from .generation import NOISE_FLOOR, _FrameContext, _load_states
from .generation import _swing_scale as _generation_swing_scale
from .models.data import Stream  # noqa: F401  re-exported: defined here before the models package
from .registry import AssetSpec
from .timeline import (
    _benign_gap,
    _ramp_episode,
    _single_shot_episode,
    _TimelineBuffers,
    _want_attack,
)


@dataclass
class _StreamPlan:
    """How a stream is walked: which families rotate, whether the ramp is among them, the ramp
    shape, and the attacked fraction the gaps are sized for. The published streams' scheduler:
    families drawn uniformly, the ramp with probability 1/(n+1), the stream episode-length bands."""

    single: list[int]  # single-shot families (persist as a flat episode)
    has_ramp: bool
    ramp_len: int
    ramp_rate: float
    attacked_frac: float


def _advance(
    ctx: _FrameContext, buf: _TimelineBuffers, rng: np.random.Generator, t: int, plan: _StreamPlan
) -> int:
    """One step of the timeline walk: a benign gap when the attacked fraction is on target (or
    nothing can attack), else a ramp episode or a single-shot episode. Returns the next free timestep."""
    if not _want_attack(buf, t, plan.attacked_frac) or not (plan.single or plan.has_ramp):
        return _benign_gap(ctx, buf, rng, t)
    use_ramp = plan.has_ramp and (not plan.single or rng.random() < 1.0 / (len(plan.single) + 1))
    if use_ramp and t < len(ctx.X) - plan.ramp_len:
        return _ramp_episode(ctx, buf, rng, t, plan.ramp_len, plan.ramp_rate)
    fid = int(rng.choice(plan.single)) if plan.single else RAMP_FAMILY
    return _single_shot_episode(ctx, buf, rng, t, fid)


def _stream_result(
    g: FdiaGenerator, X: np.ndarray, buf: _TimelineBuffers, T: int, out: Optional[str]
) -> Stream:
    """Assemble the stream dict (and save it when `out` is given)."""
    # clean = the NOISELESS healthy state at every timestep (the truth the attack was injected onto), in the
    # same column order as node_x ([|V|, Pinj, Qinj, angle]). Three aligned layers per frame: node_x
    # (attacked+noisy observed) -> benign (attack removed, noise kept) -> clean (noise removed too).
    clean = X[:T].astype(np.float32)
    # Static graph for PyG-style models: edge_index [2,E] connectivity (int64 -> torch.long) + edge_attr
    # [E,8] line features (r, x, b, g, series-admittance gs/bs, tap, shift). Same every frame.
    edge_index = np.asarray(g.ei, dtype=np.int64)
    edge_attr = np.stack(
        [
            g.branch.r,
            g.branch.x,
            g.branch.b,
            g.branch.g,
            g.edge_gs,
            g.edge_bs,
            g.branch.tap,
            g.branch.shift_deg,
        ],
        axis=1,
    ).astype(np.float32)
    # Static availability masks (which channels carry a meter), the same sparse plan every frame.
    masks = g.emit_from_state(X[0])  # the same sparse plan every frame
    result: dict[str, Any] = dict(
        node_x=buf.node_x,
        benign=buf.benign,
        clean=clean,
        edge_x=buf.edge_x,
        edge_benign=buf.edge_benign,
        edge_clean=buf.edge_clean,
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_m=masks.node_m.astype(np.uint8),
        edge_m=masks.edge_m.astype(np.uint8),
        y=buf.y,
        family=buf.family,
        temporal_delta=buf.temporal_delta,
        swing=buf.swing,
        timestep=np.arange(T),
        episodes=buf.episodes,
    )
    if out:
        np.savez_compressed(out, **{**result, "episodes": np.array(buf.episodes, dtype=object)})
    return Stream(**result, **stream_summary(result))


def stream_summary(s: dict[str, Any]) -> dict[str, Any]:
    """The two summary fields of a stream, derived from its arrays: `system` is the bus count and
    `attacked_frac` the fraction of frames with at least one attacked bus. `load_stream` uses this
    because the stream files carry the arrays only."""
    y = np.asarray(s["y"])
    return {
        "system": int(np.asarray(s["node_x"]).shape[1]),
        "attacked_frac": float((y.sum(axis=1) > 0).mean()),
    }


def generate_stream(
    system: Union[int, str],
    states: Optional[Union[str, np.ndarray]] = None,
    attacked_frac: float = 0.5,
    families: Sequence[str] = ("Aq", "Ad", "As", "Ar", "At", "Al"),
    attack_intensity: float = 0.20,
    ramp_rate: float = 0.002,
    ramp_len: int = 60,
    replay_tau: Optional[int] = None,
    redundancy: Optional[dict] = None,
    seed: int = 123,
    out: Optional[str] = None,
) -> Stream:
    """Build one continuous attacked time series for `system`. Returns a dict (also saved to `out` if given).

    attacked_frac : target fraction of timesteps under an attack episode (~0.5 = balanced).
    families      : attack families to rotate through; "At" is the slow ramp (its own episode shape).
    Other knobs mirror `generate`. Reuses the exact per-frame attack physics so streamed attacks match the shard.
    """
    red = {"vbus_frac": 0.6, "pmu_frac": 0.2, "flow_frac": 0.9, **(redundancy or {})}
    g = FdiaGenerator(system, seed=seed, **red)
    lra_k = min(6, len(g.load_bus))
    g._pick_lra_target(attack_intensity, lra_k, n_targets=15)
    X = _load_states(system, states)
    T, C = len(X), g.C
    knobs = FrameKnobs(
        attack_intensity, NOISE_FLOOR, lra_k, replay_tau, reject_below_floor=False, with_benign=True
    )
    ctx = _FrameContext(g, X, _generation_swing_scale(X, C), knobs, [])
    fam_ids = [FAM_ID[f] for f in families]
    plan = _StreamPlan(
        [f for f in fam_ids if f in SINGLE_SHOT_ORDER],
        RAMP_FAMILY in fam_ids,
        ramp_len,
        ramp_rate,
        attacked_frac,
    )
    # All noiseless from-end flows over the whole timeline in one batched matmul (metered branches only, so
    # edge_benign - edge_clean is the meter error on the measured channels). Shared physics primitive.
    buf = _TimelineBuffers(T, C, g.E, ctx.scale, g.clean_flows_from_states(X[:T]))
    # Walk the timeline: alternate a benign gap and an attack episode, sized so the attacked fraction ~ target.
    t = 0
    while t < T:
        t = _advance(ctx, buf, g.rng, t, plan)
    return _stream_result(g, X, buf, T, out)


_GRAPH_KEYS = ("edge_index", "edge_attr", "node_m", "edge_m")  # PyG-ready graph + static meter masks


def _asset_spec(name: str, file: str, release: Optional[str]) -> AssetSpec:
    from .registry import _REPO, STREAM_RELEASE

    return AssetSpec("builtin", name, file=file, release=release or STREAM_RELEASE, repo=_REPO)


def _attach_graph_sidecar(out: dict[str, Any], C: int, release: Optional[str]) -> None:
    """Newer streams embed the graph and masks; streams that predate them (e.g. the v0.7.1 assets) get
    the tiny per-system graph sidecar, so every load_stream dict is complete."""
    from .download import ensure_local

    if all(k in out for k in _GRAPH_KEYS):
        return
    gz = np.load(ensure_local(_asset_spec(f"graph{C}", f"graph_ieee{C}.npz", release)))
    for k in _GRAPH_KEYS:
        if k not in out and k in gz.files:
            out[k] = gz[k]


def _normalize_graph_dtypes(out: dict[str, Any]) -> None:
    """The same dtypes whatever the source: edge_index int64 (torch.long), edge_attr float32, meter
    masks uint8 (as generate_stream writes them), so embedded and sidecar loads are identical."""
    if "edge_index" in out:
        out["edge_index"] = np.asarray(out["edge_index"], dtype=np.int64)
    if "edge_attr" in out:
        out["edge_attr"] = np.asarray(out["edge_attr"], dtype=np.float32)
    for m in ("node_m", "edge_m"):
        if m in out:
            out[m] = np.asarray(out[m], dtype=np.uint8)


def load_stream(system: Union[int, str], release: Optional[str] = None) -> Stream:
    """Download (and cache) the published continuous stream for a system and return it as a dict.

    Same dict shape as generate_stream (node_x, benign, clean, edge_x/edge_benign/edge_clean, y, family, ...).
    Built-in systems only (14/30/57/89/118/145/200/300). release: None -> newest published streams; a tag pins
    a version. Streams ship in the same complete release as the shards (STREAM_RELEASE follows _RELEASE), so this
    tracks the latest continuous-dataset release without disturbing which shard release fg.load() uses.
    """
    from .download import ensure_local
    from .registry import system_id

    C = system_id(system)
    z = np.load(ensure_local(_asset_spec(f"stream{C}", f"stream_ieee{C}.npz", release)), allow_pickle=True)
    out = {k: z[k] for k in z.files}
    _attach_graph_sidecar(out, C, release)
    _normalize_graph_dtypes(out)
    return Stream(**out, **{k: v for k, v in stream_summary(out).items() if k not in out})


def _check_window_args(T: int, W: int, stride: int, label: str) -> None:
    """Reject a window request the docstring of `windows` does not allow, before any slicing."""
    if label not in ("frame", "any", "last"):
        raise ValueError(f"label must be 'frame', 'any' or 'last', got {label!r}")
    integral = all(isinstance(v, numbers.Integral) and not isinstance(v, bool) for v in (W, stride))
    if not integral or not 1 <= W <= T or stride < 1:
        raise ValueError(
            f"need integers 1 <= W <= {T} frames and stride >= 1, got W={W!r}, stride={stride!r}"
        )


def _window_labels(y: np.ndarray, starts: range, W: int, label: str) -> np.ndarray:
    """Per-window labels: every frame ("frame"), attacked at any frame ("any"), or the last frame."""
    if label == "frame":
        return np.stack([y[s : s + W] for s in starts])
    if label == "last":
        return np.stack([y[s + W - 1] for s in starts])
    return np.stack([y[s : s + W].max(0) for s in starts])


def windows(
    stream: dict[str, Any], W: int, stride: int = 1, label: str = "any"
) -> tuple[np.ndarray, np.ndarray]:
    """Slide a length-W window over a stream. Returns (Xw [n,W,N,4], yw).

    label: "frame" -> per-frame per-bus labels yw [n,W,N]; "any" -> window-level per-bus label yw [n,N]
    (bus attacked at ANY frame in the window); "last" -> label at the final frame yw [n,N].
    """
    nx = stream["node_x"]
    y = stream["y"]
    T = len(nx)
    _check_window_args(T, W, stride, label)
    starts = range(0, T - W + 1, stride)
    Xw = np.stack([nx[s : s + W] for s in starts])
    return Xw, _window_labels(y, starts, W, label)
