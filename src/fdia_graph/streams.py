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

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import warnings

import numpy as np

from .engine import FdiaGenerator, FAM_ID
from .engine.records import RAMP_FAMILY, SINGLE_SHOT_ORDER, Frame, FrameKnobs, attack_frame
from .formulas.attacks import ramp_profile
from .formulas.temporal import swing_zscore, temporal_delta
from .generation import _FrameContext, _load_states, NOISE_FLOOR
from .generation import _swing_scale as _generation_swing_scale
from .registry import AssetSpec

# Per-family episode-length band (frames). Ramp spans its full ramp_len; spike/measurement/redistribution
# families persist for a shorter, variable window. Benign gaps are drawn from the same overall scale so the
# attacked fraction lands near the requested target.
_EP_LEN = {1: (15, 45), 2: (5, 25), 3: (5, 25), 4: (5, 25), 6: (10, 30)}  # Aq, Ad, As, Ar, Al


def _swing_scale(X: np.ndarray, C: int) -> np.ndarray:
    """Deprecated alias: the swing scale moved to fdia_graph.generation._swing_scale (shared by shards
    and streams). Removed one minor version after 0.16."""
    warnings.warn(
        "fdia_graph.streams._swing_scale moved to fdia_graph.generation._swing_scale",
        DeprecationWarning,
        stacklevel=2,
    )
    return _generation_swing_scale(X, C)


class _StreamBuffers:
    """The per-frame layers of one stream, allocated once and filled frame by frame.

    node_x / edge_x are the OBSERVED measurements (attacked + noisy where attacked, else benign + noisy);
    benign / edge_benign the same scan with the attack removed and the noise kept; edge_clean the
    noiseless true flows on metered branches. temporal_delta and swing are the two temporal features,
    taken against the previous EMITTED frame (the stream has no per-record pool lookup).
    """

    def __init__(self, T: int, C: int, E: int, scale: np.ndarray, edge_clean_full: np.ndarray) -> None:
        self.node_x = np.zeros((T, C, 4), np.float32)
        self.benign = np.zeros((T, C, 4), np.float32)
        self.edge_x = np.zeros((T, E, 2), np.float32)
        self.edge_benign = np.zeros((T, E, 2), np.float32)
        self.edge_clean = np.zeros((T, E, 2), np.float32)
        self.y = np.zeros((T, C), np.uint8)
        self.family = np.zeros(T, np.int16)
        self.temporal_delta = np.zeros((T, C, 2), np.float32)
        self.swing = np.zeros((T, C, 2), np.float32)
        self.episodes: List[Dict[str, Any]] = []
        self._scale = scale
        self._edge_clean_full = edge_clean_full
        self._prev_nx: Optional[np.ndarray] = None
        self._all_buses = np.ones(C, bool)

    def store(
        self,
        t: int,
        nx: np.ndarray,
        yt: np.ndarray,
        fid: int,
        benign_nx: np.ndarray,
        ex: np.ndarray,
        benign_ex: np.ndarray,
    ) -> None:
        self.node_x[t] = nx
        self.benign[t] = benign_nx
        self.y[t] = yt
        self.family[t] = fid
        self.edge_x[t] = ex
        self.edge_benign[t] = benign_ex
        self.edge_clean[t] = self._edge_clean_full[t]
        # The two temporal features against the previous EMITTED frame, through the same kernel the
        # shard uses; the stream computes them at every bus (an unmetered bus reads 0 - 0).
        p = self._prev_nx if self._prev_nx is not None else nx
        self.temporal_delta[t] = temporal_delta(nx, p, self._all_buses)
        self.swing[t] = swing_zscore(nx, p, self._scale[t], self._all_buses)
        self._prev_nx = nx

    def store_benign(self, t: int, nx: np.ndarray, ex: np.ndarray) -> None:
        """A benign frame: observed == un-attacked."""
        self.store(t, nx, np.zeros(self.y.shape[1], np.uint8), 0, nx, ex, ex)

    def store_frame(self, t: int, fid: int, frame: Frame) -> None:
        """An attacked frame with its un-attacked twin; stream frames set with_benign=True, so both exist."""
        assert frame.benign_node_x is not None and frame.benign_edge_x is not None
        self.store(t, frame.node_x, frame.y, fid, frame.benign_node_x, frame.edge_x, frame.benign_edge_x)


@dataclass
class _StreamPlan:
    """How the timeline is walked: which families rotate, whether the ramp is among them, the
    ramp shape, and the attacked fraction the gaps are sized for."""

    single: List[int]  # single-shot families (persist as a flat episode)
    has_ramp: bool
    ramp_len: int
    ramp_rate: float
    attacked_frac: float


def _emit_benign(ctx: _FrameContext, t: int) -> Tuple[np.ndarray, np.ndarray]:
    """The benign scan of timestep t (also remembered for the replay families)."""
    frame = attack_frame(ctx.g, ctx.X[t], 0, None, None, ctx.knobs)
    assert frame is not None  # a benign emission cannot fail
    return frame.node_x, frame.edge_x


def _pick_targets(rng: np.random.Generator, apos: np.ndarray, fid: int) -> np.ndarray:
    """Attacked load-table positions for a stream episode: 1 to 6 buses for Aq, up to 4 otherwise."""
    nab = len(apos)
    k = int(rng.integers(1, min(6, nab) + 1)) if fid == 1 else min(4, nab)
    return rng.choice(apos, k, replace=False)


def _want_attack(y: np.ndarray, t: int, attacked_frac: float) -> bool:
    """Start an attack episode when the attacked fraction so far is below the target."""
    if t == 0:
        return True
    atk_so_far = int((y[:t].sum(axis=1) > 0).sum())  # frames with any attacked bus, so far
    return (atk_so_far / max(1, t)) < attacked_frac


def _benign_gap(ctx: _FrameContext, buf: _StreamBuffers, rng: np.random.Generator, t: int) -> int:
    """A benign gap of 5 to 39 frames; returns the next free timestep."""
    T = len(ctx.X)
    gap = int(rng.integers(5, 40))
    for _ in range(gap):
        if t >= T:
            break
        bn, bex = _emit_benign(ctx, t)
        buf.store_benign(t, bn, bex)
        t += 1
    return t


def _store_or_benign(
    ctx: _FrameContext, buf: _StreamBuffers, t: int, fid: int, frame: Optional[Frame], ok: np.ndarray
) -> None:
    """Store the attacked frame, or a benign one when the attack could not be built at this timestep."""
    if frame is None:
        bn, bex = _emit_benign(ctx, t)
        buf.store_benign(t, bn, bex)
    else:
        buf.store_frame(t, fid, frame)
        ok |= frame.y


def _ramp_episode(
    ctx: _FrameContext, buf: _StreamBuffers, rng: np.random.Generator, t: int, plan: _StreamPlan
) -> int:
    """One slow-ramp episode on a fixed bus set (rise, hold, return); returns the next free timestep."""
    T, C = len(ctx.X), ctx.g.C
    apos = ctx.g.attackable_pos
    a = rng.choice(apos, min(5, len(apos)), replace=False)  # fixed bus set for the ramp
    direction = 1.0 if rng.random() < 0.5 else -1.0
    rise = max(1, int(rng.uniform(0.2, 0.45) * plan.ramp_len))
    hold = int(rng.uniform(0.0, 0.25) * plan.ramp_len)
    t0 = t
    ok = np.zeros(C, np.uint8)
    for i in range(plan.ramp_len):
        if t >= T:
            break
        dev = ramp_profile(i, rise, hold, plan.ramp_rate, plan.ramp_rate)
        frame = attack_frame(ctx.g, ctx.X[t], RAMP_FAMILY, a, 1 + direction * dev, ctx.knobs)
        _store_or_benign(ctx, buf, t, RAMP_FAMILY, frame, ok)
        t += 1
    buf.episodes.append(dict(onset=t0, length=t - t0, family=RAMP_FAMILY, buses=np.where(ok)[0].tolist()))
    return t


def _single_shot_episode(
    ctx: _FrameContext, buf: _StreamBuffers, rng: np.random.Generator, t: int, fid: int
) -> int:
    """One episode of a single-shot family held for a random length; returns the next free timestep."""
    T, C = len(ctx.X), ctx.g.C
    a = _pick_targets(rng, ctx.g.attackable_pos, fid)
    mult = 1 + rng.uniform(0.05, ctx.knobs.intensity, size=len(a))
    L = int(rng.integers(*_EP_LEN.get(fid, (5, 25))))
    t0 = t
    ok = np.zeros(C, np.uint8)
    for _ in range(L):
        if t >= T:
            break
        _store_or_benign(ctx, buf, t, fid, attack_frame(ctx.g, ctx.X[t], fid, a, mult, ctx.knobs), ok)
        t += 1
    buf.episodes.append(dict(onset=t0, length=t - t0, family=fid, buses=np.where(ok)[0].tolist()))
    return t


def _advance(
    ctx: _FrameContext, buf: _StreamBuffers, rng: np.random.Generator, t: int, plan: _StreamPlan
) -> int:
    """One step of the timeline walk: a benign gap when the attacked fraction is on target (or
    nothing can attack), else a ramp episode or a single-shot episode. Returns the next free timestep."""
    if not _want_attack(buf.y, t, plan.attacked_frac) or not (plan.single or plan.has_ramp):
        return _benign_gap(ctx, buf, rng, t)
    use_ramp = plan.has_ramp and (not plan.single or rng.random() < 1.0 / (len(plan.single) + 1))
    if use_ramp and t < len(ctx.X) - plan.ramp_len:
        return _ramp_episode(ctx, buf, rng, t, plan)
    fid = int(rng.choice(plan.single)) if plan.single else RAMP_FAMILY
    return _single_shot_episode(ctx, buf, rng, t, fid)


def _stream_result(
    g: FdiaGenerator, X: np.ndarray, buf: _StreamBuffers, T: int, out: Optional[str]
) -> Dict[str, Any]:
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
    result: Dict[str, Any] = dict(
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
    result["system"] = g.C
    result["attacked_frac"] = float((buf.y.sum(axis=1) > 0).mean())
    return result


def generate_stream(
    system: Union[int, str],
    states: Optional[Union[str, np.ndarray]] = None,
    attacked_frac: float = 0.5,
    families: Sequence[str] = ("Aq", "Ad", "As", "Ar", "At", "Al"),
    attack_intensity: float = 0.20,
    ramp_rate: float = 0.002,
    ramp_len: int = 60,
    replay_tau: Optional[int] = None,
    redundancy: Optional[Dict] = None,
    seed: int = 123,
    out: Optional[str] = None,
) -> Dict[str, Any]:
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
    buf = _StreamBuffers(T, C, g.E, ctx.scale, g.clean_flows_from_states(X[:T]))
    # Walk the timeline: alternate a benign gap and an attack episode, sized so the attacked fraction ~ target.
    t = 0
    while t < T:
        t = _advance(ctx, buf, g.rng, t, plan)
    return _stream_result(g, X, buf, T, out)


_GRAPH_KEYS = ("edge_index", "edge_attr", "node_m", "edge_m")  # PyG-ready graph + static meter masks


def _asset_spec(name: str, file: str, release: Optional[str]) -> AssetSpec:
    from .registry import _REPO, STREAM_RELEASE

    return AssetSpec("builtin", name, file=file, release=release or STREAM_RELEASE, repo=_REPO)


def _attach_graph_sidecar(out: Dict[str, Any], C: int, release: Optional[str]) -> None:
    """Newer streams embed the graph and masks; streams that predate them (e.g. the v0.7.1 assets) get
    the tiny per-system graph sidecar, so every load_stream dict is complete."""
    from .download import ensure_local

    if all(k in out for k in _GRAPH_KEYS):
        return
    gz = np.load(ensure_local(_asset_spec(f"graph{C}", f"graph_ieee{C}.npz", release)))
    for k in _GRAPH_KEYS:
        if k not in out and k in gz.files:
            out[k] = gz[k]


def _normalize_graph_dtypes(out: Dict[str, Any]) -> None:
    """The same dtypes whatever the source: edge_index int64 (torch.long), edge_attr float32, meter
    masks uint8 (as generate_stream writes them), so embedded and sidecar loads are identical."""
    if "edge_index" in out:
        out["edge_index"] = np.asarray(out["edge_index"], dtype=np.int64)
    if "edge_attr" in out:
        out["edge_attr"] = np.asarray(out["edge_attr"], dtype=np.float32)
    for m in ("node_m", "edge_m"):
        if m in out:
            out[m] = np.asarray(out[m], dtype=np.uint8)


def load_stream(system: Union[int, str], release: Optional[str] = None) -> Dict[str, Any]:
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
    return out


def windows(
    stream: Dict[str, Any], W: int, stride: int = 1, label: str = "any"
) -> Tuple[np.ndarray, np.ndarray]:
    """Slide a length-W window over a stream. Returns (Xw [n,W,N,4], yw).

    label: "frame" -> per-frame per-bus labels yw [n,W,N]; "any" -> window-level per-bus label yw [n,N]
    (bus attacked at ANY frame in the window); "last" -> label at the final frame yw [n,N].
    """
    nx = stream["node_x"]
    y = stream["y"]
    T = len(nx)
    starts = range(0, T - W + 1, stride)
    Xw = np.stack([nx[s : s + W] for s in starts])
    if label == "frame":
        yw = np.stack([y[s : s + W] for s in starts])
    elif label == "last":
        yw = np.stack([y[s + W - 1] for s in starts])
    else:
        yw = np.stack([y[s : s + W].max(0) for s in starts])
    return Xw, yw
