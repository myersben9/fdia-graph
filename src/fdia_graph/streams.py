"""Continuous attacked streams — the realistic time series for temporal models (LSTM / TGN).

The classification shard (`generate`) is a SHUFFLED table of independent labeled snapshots: attacks are
injected at scattered timesteps and mixed together, so its rows are not contiguous in time. A temporal model
wants the opposite: one running timeline where the grid operates normally and an attack appears over a
contiguous EPISODE, then clears. `generate_stream` produces exactly that, per system:

    Three aligned measurement layers per frame ([|V|, Pinj, Qinj, angle] columns):
    node_x   [T, N, 4]  OBSERVED feed — attacked+noisy where attacked, benign+noisy elsewhere (the model input)
    benign   [T, N, 4]  the same meters with the ATTACK REMOVED (benign+noisy) — what they would read un-attacked
    clean    [T, N, 4]  NOISELESS attack-free TRUE state — the SE / reconstruction target
                        (node_x == benign on benign frames; node_x - benign is the attack; benign - clean is meter noise)
    edge_x, edge_benign, edge_clean [T, E, 2]  the SAME three layers for branch flows [P_from, Q_from]
                        (observed / attack-removed / noiseless). Node + edge together are the full SE measurement set.
    y        [T, N]      per-timestep, per-bus attack label (0 on benign frames/buses)
    family   [T]         active attack family id at each timestep (0 = benign)
    temporal_delta, swing [T, N, 2]   change vs the PREVIOUS EMITTED frame (see note below)
    episodes list        (onset, length, family, attacked buses) for every attack episode

Temporal-feature note: unlike the shard (which compares an attacked snapshot to the benign X[t-1]), a stream
compares each frame to the previous EMITTED frame. That is what keeps a stealthy ramp looking like a small
per-step change and a spike looking like an abrupt jump — the spike-vs-ramp signal the dataset is built on.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ._core import FdiaGenerator, FAM_ID
from .generation import _load_states, SWING_W, NOISE_FLOOR, _FAMK

# Per-family episode-length band (frames). Ramp spans its full ramp_len; spike/measurement/redistribution
# families persist for a shorter, variable window. Benign gaps are drawn from the same overall scale so the
# attacked fraction lands near the requested target.
_EP_LEN = {1: (15, 45), 2: (5, 25), 3: (5, 25), 4: (5, 25), 6: (10, 30)}   # Aq, Ad, As, Ar, Al


def _swing_scale(X: np.ndarray, C: int) -> np.ndarray:
    """Per-timestep benign 'typical recent change' std over the last SWING_W scans (prefix-sum, same as the shard)."""
    T = len(X)
    D = np.abs(np.diff(X[:, :, :2], axis=0))                              # [T-1,N,2] scan-to-scan |change|
    c1 = np.concatenate([np.zeros((1,) + D.shape[1:]), np.cumsum(D, 0)], 0)
    c2 = np.concatenate([np.zeros((1,) + D.shape[1:]), np.cumsum(D ** 2, 0)], 0)
    SCALE = np.full((T, C, 2), 1e-3, np.float32)
    for t in range(2, T):
        s = max(0, t - SWING_W); e = t - 1; n = e - s
        if n >= 3:
            su = c1[e] - c1[s]; sq = c2[e] - c2[s]
            SCALE[t] = np.sqrt(np.maximum(sq / n - (su / n) ** 2, 0.0)) + 1e-3
    return SCALE


def generate_stream(system: Union[int, str], states: Optional[Union[str, np.ndarray]] = None,
                    attacked_frac: float = 0.5, families: Sequence[str] = ("Aq", "Ad", "As", "Ar", "At", "Al"),
                    attack_intensity: float = 0.20, ramp_rate: float = 0.002, ramp_len: int = 60,
                    replay_tau: Optional[int] = None, redundancy: Optional[Dict] = None, seed: int = 123,
                    out: Optional[str] = None) -> Dict[str, Any]:
    """Build one continuous attacked time series for `system`. Returns a dict (also saved to `out` if given).

    attacked_frac : target fraction of timesteps under an attack episode (~0.5 = balanced).
    families      : attack families to rotate through; "At" is the slow ramp (its own episode shape).
    Other knobs mirror `generate`. Reuses the exact per-frame attack physics so streamed attacks match the shard.
    """
    red = {"vbus_frac": 0.6, "pmu_frac": 0.2, "flow_frac": 0.9, **(redundancy or {})}
    g = FdiaGenerator(system, seed=seed, **red)
    g._pick_lra_target(attack_intensity, min(6, len(g.load_bus)), n_targets=15)
    K = min(6, len(g.load_bus)); rng = g.rng
    X = _load_states(system, states); T = len(X); C = g.C
    SCALE = _swing_scale(X, C)
    apos = g.attackable_pos
    fam_ids = [FAM_ID[f] for f in families]
    single = [f for f in fam_ids if f in (1, 2, 3, 4, 6)]   # single-shot families (persist as a flat episode)
    has_ramp = 5 in fam_ids

    E = g.E
    node_x = np.zeros((T, C, 4), np.float32)      # OBSERVED node feed: attacked+noisy where attacked, else benign+noisy
    benign = np.zeros((T, C, 4), np.float32)      # UN-ATTACKED node measurement (benign+noisy), attack removed
    edge_x = np.zeros((T, E, 2), np.float32)      # OBSERVED branch flows [P_from, Q_from] (attacked+noisy / benign+noisy)
    edge_ben = np.zeros((T, E, 2), np.float32)    # UN-ATTACKED branch flows (benign+noisy)
    edge_cln = np.zeros((T, E, 2), np.float32)    # NOISELESS true branch flows
    y = np.zeros((T, C), np.uint8)
    fam = np.zeros(T, np.int16)
    td_all = np.zeros((T, C, 2), np.float32)
    sw_all = np.zeros((T, C, 2), np.float32)
    episodes: List[Dict[str, Any]] = []
    prev_nx: Optional[np.ndarray] = None

    def _clean_edges(xt: np.ndarray) -> np.ndarray:
        # Noiseless from-end branch flow Sf = V_from * conj(Yf @ V) * baseMVA, straight from the true state.
        Vc = np.zeros(g._nppc, complex)
        for b in range(C): Vc[g._lut[b]] = xt[b, 2] * np.exp(1j * np.deg2rad(xt[b, 3]))
        Sf = Vc[g._fb] * np.conj(g._Yf @ Vc) * g._bMVA
        return np.stack([Sf.real, Sf.imag], axis=1).astype(np.float32)

    def _emit_benign(t: int) -> Tuple[np.ndarray, np.ndarray]:
        nx, nm, ex, em = g.emit_from_state(X[t])
        g.benign_buf.append(nx.copy())
        if len(g.benign_buf) > 300: g.benign_buf.pop(0)
        return nx, ex

    def _store(t: int, nx: np.ndarray, yt: np.ndarray, fid: int, benign_nx: np.ndarray,
               ex: np.ndarray, benign_ex: np.ndarray) -> None:
        nonlocal prev_nx
        node_x[t] = nx; benign[t] = benign_nx; y[t] = yt; fam[t] = fid
        edge_x[t] = ex; edge_ben[t] = benign_ex; edge_cln[t] = _clean_edges(X[t])
        p = prev_nx if prev_nx is not None else nx
        td_all[t, :, 0] = nx[:, 1] - p[:, 1]; td_all[t, :, 1] = nx[:, 2] - p[:, 2]   # vs previous EMITTED frame
        sc = SCALE[t]; sw_all[t, :, 0] = td_all[t, :, 0] / sc[:, 0]; sw_all[t, :, 1] = td_all[t, :, 1] / sc[:, 1]
        prev_nx = nx

    def _attack_frame(t: int, fid: int, a: np.ndarray, mult: Union[float, np.ndarray]
                      ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Apply family `fid` at timestep t. Returns (nx, y, benign_nx, ex, benign_ex) or None, where ex is the
        OBSERVED branch flows and benign_ex the un-attacked branch flows (node + edge from the same scan)."""
        yt = np.zeros(C, np.uint8)
        if fid in (1, 5):                                    # Aq / ramp: re-solve with scaled load
            Lp = X[t][g.load_bus, 0] + g.load_genP; Lq = X[t][g.load_bus, 1].copy(); Lp_true = Lp.copy()
            Lp = Lp.copy(); Lp[a] *= mult
            net = g.solve(Lp, Lq, Xt=X[t], Lp_true=Lp_true)
            if net is None: return None
            nx, nm, ex, em = g.emit(net); yt[g.load_bus[a]] = 1
            bnx, bnm, bex, bem = g.emit_from_state(X[t])     # benign = un-attacked emit of the true state
            return nx, yt, bnx, ex, bex
        if fid == 6:                                         # Al / LRA: load-redistribution re-solve
            Lp = X[t][g.load_bus, 0] + g.load_genP; Lq = X[t][g.load_bus, 1].copy()
            d, aa = g.lra_delta(Lp, attack_intensity, K, floor=NOISE_FLOOR)
            if len(aa) == 0: return None
            net = g.solve(Lp + d, Lq, Xt=X[t], Lp_true=Lp)
            if net is None: return None
            nx, nm, ex, em = g.emit(net); yt[g.load_bus[aa]] = 1
            bnx, bnm, bex, bem = g.emit_from_state(X[t])
            return nx, yt, bnx, ex, bex
        nx, nm, ex, em = g.emit_from_state(X[t])             # Ad/As/Ar: corrupt measurements in place
        benign_nx = nx.copy(); benign_ex = ex.copy()         # capture BEFORE corruption -> shares noise with nx/ex
        abus = g.load_bus[a]
        if replay_tau is not None and g.benign_buf:          # fixed replay depth, else random lag >=20
            replay = g.benign_buf[-min(replay_tau, len(g.benign_buf))]
        elif len(g.benign_buf) > 20:
            replay = g.benign_buf[int(rng.integers(0, len(g.benign_buf) - 20))]
        else:
            replay = g.benign_buf[0] if g.benign_buf else None
        nx, ex, weak, mags = g.corrupt(nx, ex, abus, _FAMK[fid], replay, floor=NOISE_FLOOR, cap=attack_intensity)
        yt[abus] = 1
        return nx, yt, benign_nx, ex, benign_ex             # un-attacked node/edge in benign == nx/ex exactly

    def _pick_targets(fid: int) -> np.ndarray:
        nab = len(apos)
        k = int(rng.integers(1, min(6, nab) + 1)) if fid == 1 else min(4, nab)
        return rng.choice(apos, k, replace=False)

    # Walk the timeline: alternate a benign gap and an attack episode, sized so the attacked fraction ~ target.
    t = 0
    while t < T:
        atk_so_far = int((y[:t].sum(axis=1) > 0).sum())   # frames with any attacked bus, so far
        want_attack = (atk_so_far / max(1, t)) < attacked_frac if t > 0 else True
        if not want_attack or not (single or has_ramp):
            gap = int(rng.integers(5, 40))
            for _ in range(gap):
                if t >= T: break
                bn, bex = _emit_benign(t); _store(t, bn, np.zeros(C, np.uint8), 0, bn, bex, bex); t += 1   # benign: observed == un-attacked
            continue
        # start an attack episode
        use_ramp = has_ramp and (not single or rng.random() < 1.0 / (len(single) + 1))
        if use_ramp and t < T - ramp_len:
            a = rng.choice(apos, min(5, len(apos)), replace=False)      # fixed bus set for the ramp
            direction = 1.0 if rng.random() < 0.5 else -1.0
            rise = max(1, int(rng.uniform(0.2, 0.45) * ramp_len)); hold = int(rng.uniform(0.0, 0.25) * ramp_len)
            t0 = t; ok = np.zeros(C, np.uint8)
            for i in range(ramp_len):
                if t >= T: break
                dev = ramp_rate * (i if i < rise else (rise if i < rise + hold else max(0, rise - (i - rise - hold))))
                res = _attack_frame(t, 5, a, 1 + direction * dev)
                if res is None:
                    bn, bex = _emit_benign(t); _store(t, bn, np.zeros(C, np.uint8), 0, bn, bex, bex)
                else:
                    _store(t, res[0], res[1], 5, res[2], res[3], res[4]); ok |= res[1]
                t += 1
            episodes.append(dict(onset=t0, length=t - t0, family=5, buses=np.where(ok)[0].tolist()))
        else:
            fid = int(rng.choice(single)) if single else 5
            a = _pick_targets(fid); mult = 1 + rng.uniform(0.05, attack_intensity, size=len(a))
            L = int(rng.integers(*_EP_LEN.get(fid, (5, 25)))); t0 = t; ok = np.zeros(C, np.uint8)
            for _ in range(L):
                if t >= T: break
                res = _attack_frame(t, fid, a, mult)
                if res is None:
                    bn, bex = _emit_benign(t); _store(t, bn, np.zeros(C, np.uint8), 0, bn, bex, bex)
                else:
                    _store(t, res[0], res[1], fid, res[2], res[3], res[4]); ok |= res[1]
                t += 1
            episodes.append(dict(onset=t0, length=t - t0, family=fid, buses=np.where(ok)[0].tolist()))

    # clean = the NOISELESS healthy state at every timestep (the truth the attack was injected onto), in the
    # same column order as node_x ([|V|, Pinj, Qinj, angle]). This is the SE / reconstruction target: pair
    # (node_x[t], clean[t]) is (attacked measurements, true state) even on attacked frames.
    clean = np.stack([X[:T, :, 2], X[:T, :, 0], X[:T, :, 1], X[:T, :, 3]], axis=2).astype(np.float32)
    # Three aligned layers per frame: node_x (attacked+noisy observed) -> benign (attack removed, noise kept)
    # -> clean (noise removed too, the true state). node_x == benign on benign frames.
    # Branch-flow measurements mirror the node layers: edge_x (observed) -> edge_benign (attack removed)
    # -> edge_clean (noiseless true flows). Node + edge from the same scan gives the full SE measurement set.
    result = dict(node_x=node_x, benign=benign, clean=clean,
                  edge_x=edge_x, edge_benign=edge_ben, edge_clean=edge_cln,
                  y=y, family=fam, temporal_delta=td_all, swing=sw_all,
                  timestep=np.arange(T), system=C, episodes=episodes,
                  attacked_frac=float((y.sum(axis=1) > 0).mean()))
    if out:
        np.savez_compressed(out, node_x=node_x, benign=benign, clean=clean,
                            edge_x=edge_x, edge_benign=edge_ben, edge_clean=edge_cln,
                            y=y, family=fam, temporal_delta=td_all, swing=sw_all,
                            timestep=np.arange(T), episodes=np.array(episodes, dtype=object))
    return result


def load_stream(system: Union[int, str], release: Optional[str] = None) -> Dict[str, Any]:
    """Download (and cache) the published continuous stream for a system and return it as a dict.

    Same dict shape as generate_stream (node_x, benign, clean, edge_x/edge_benign/edge_clean, y, family, ...).
    Built-in systems only (14/30/57/89/118/145/200/300). release: None -> newest published streams; a tag pins
    a version. Streams are versioned separately from the classification shards (STREAM_RELEASE), so this
    tracks the latest continuous-dataset release without disturbing which shard release fg.load() uses.
    """
    from .download import ensure_local
    from .registry import _REPO, STREAM_RELEASE
    C = int(str(system).lower().replace("ieee", ""))
    spec = {"kind": "builtin", "name": f"stream{C}", "file": f"stream_ieee{C}.npz",
            "release": release or STREAM_RELEASE, "repo": _REPO, "sha256": None}
    z = np.load(ensure_local(spec), allow_pickle=True)
    return {k: z[k] for k in z.files}


def windows(stream: Dict[str, Any], W: int, stride: int = 1, label: str = "any") -> Tuple[np.ndarray, np.ndarray]:
    """Slide a length-W window over a stream. Returns (Xw [n,W,N,4], yw).

    label: "frame" -> per-frame per-bus labels yw [n,W,N]; "any" -> window-level per-bus label yw [n,N]
    (bus attacked at ANY frame in the window); "last" -> label at the final frame yw [n,N].
    """
    nx = stream["node_x"]; y = stream["y"]; T = len(nx)
    starts = range(0, T - W + 1, stride)
    Xw = np.stack([nx[s:s + W] for s in starts])
    if label == "frame":
        yw = np.stack([y[s:s + W] for s in starts])
    elif label == "last":
        yw = np.stack([y[s + W - 1] for s in starts])
    else:
        yw = np.stack([y[s:s + W].max(0) for s in starts])
    return Xw, yw
