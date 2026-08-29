"""State estimation on fdia-graph shards — the shared machinery behind every method class.

SEBase owns what all estimators have in common: the AC measurement model h(x) built from the
pandapower case, the chord-Newton iteration with its divergence guard, the meter weights calibrated
from benign residuals, and the slack handling (the slack bus is excluded from the state and pinned
to each record's true value from the shard's clean layer, so estimates and truth share one angle
frame). Subclasses change only the state space and the weights, mirroring the paper's protocol.

Needs torch and pandapower: pip install "fdia-graph[se]". Datasets must be v0.7.2+ shards (the
clean layer supplies the truth) loaded with units="physical" (the default).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np

if TYPE_CHECKING:
    from ..dataset import FdiaGraph

_CASE_FN = {
    14: "case14",
    30: "case30",
    57: "case57",
    89: "case89pegase",
    118: "case118",
    145: "case145",
    200: "case_illinois200",
    300: "case300",
}


def _torch():
    try:
        import torch

        return torch
    except ImportError as e:
        raise ImportError("state estimation needs torch: pip install 'fdia-graph[se]'") from e


class SEBase:
    """Weighted least squares AC state estimation, the audited baseline of the paper.

    Usage:
        est = WLS().fit(fg.load("ieee118", split="train"))
        xhat = est.estimate(test_ds)      # [n, 2(N-1)] = [theta rad | V pu] at non-slack buses
        rep  = est.score(test_ds)         # per-family angle/voltage MAE vs the clean truth

    fit() calibrates per-meter error scales as the rms of benign residuals at the true state
    (the accuracy-class total error, bias included) and freezes the Jacobian at the benign mean
    state, the chord that makes per-record solves cheap.
    """

    def __init__(self, npass: int = 40, iters: int = 8) -> None:
        self.npass = npass  # reweighting passes (run to convergence per the paper protocol)
        self.iters = iters  # chord-Newton steps inside each solve

    # ---- network + measurement model -------------------------------------------------------
    def _build_network(self, ds: "FdiaGraph") -> None:
        torch = _torch()
        try:
            import pandapower as pp
            import pandapower.networks as pn
            from pandapower.pypower.makeYbus import makeYbus
        except ImportError as e:
            raise ImportError("state estimation needs pandapower: pip install 'fdia-graph[se]'") from e
        if ds.units != "physical":
            raise ValueError("fit/estimate expect units='physical' datasets (the default)")
        if not ds.has_clean:
            raise ValueError("dataset has no clean layer; upgrade to a v0.7.2+ shard")
        net = getattr(pn, _CASE_FN[int(ds.system)])()
        pp.runpp(net)
        ppc = net._ppc
        Yb, Yf, _ = makeYbus(ppc["baseMVA"], ppc["bus"], ppc["branch"])
        self.baseMVA = float(ppc["baseMVA"])
        self.N = int(ds.N)
        self.E = int(ds.E)
        self._nppc = ppc["bus"].shape[0]
        self._lut = net._pd2ppc_lookups["bus"][: self.N].astype(np.int64)
        self._fb = ppc["branch"][:, 0].real.astype(np.int64)
        self._Ybus = torch.tensor(np.asarray(Yb.todense()), dtype=torch.complex128)
        self._Yft = torch.tensor(np.asarray(Yf.todense()), dtype=torch.complex128)
        self.slack = int(net.ext_grid.bus.values[0])
        self.keep = np.array([i for i in range(self.N) if i != self.slack])
        self.SD = 2 * len(self.keep)  # state dim: [theta | V] at non-slack buses
        # measurement mask, constant across records: [V(N), P(N), Q(N), theta(N), Pf(E), Qf(E)]
        nm = ds[0]["node_m"].numpy().astype(bool)
        em = ds[0]["edge_m"].numpy().astype(bool)
        self.mask = np.concatenate([nm[:, 0], nm[:, 1], nm[:, 2], nm[:, 3], em[:, 0], em[:, 1]])
        self.m = int(self.mask.sum())

    def _h(self, x: np.ndarray, vsl: np.ndarray, thsl: np.ndarray) -> np.ndarray:
        """Masked measurement prediction for a batch of states (slack pinned per record)."""
        torch = _torch()
        with torch.no_grad():
            return self._h_t(
                torch.tensor(x, dtype=torch.float64),
                torch.tensor(vsl, dtype=torch.float64),
                torch.tensor(thsl, dtype=torch.float64),
            ).numpy()[:, self.mask]

    def _h_t(self, x: Any, vsl: Any, thsl: Any) -> Any:
        # Full AC forward map (torch, unmasked). Shard injections are load-positive, so bus
        # injections are emitted as -S; flows are from-end. Exactly the paper's h.
        torch = _torch()
        B, N, ns = x.shape[0], self.N, len(self.keep)
        vsl = vsl.to(x.dtype)
        thsl = thsl.to(x.dtype)
        idx = torch.tensor(self.keep).unsqueeze(0).expand(B, ns)
        sidx = torch.full((B, 1), self.slack, dtype=torch.long)
        th = torch.zeros(B, N, dtype=x.dtype).scatter(1, idx, x[:, :ns]).scatter(1, sidx, thsl.reshape(B, 1))
        V = torch.zeros(B, N, dtype=x.dtype).scatter(1, idx, x[:, ns:]).scatter(1, sidx, vsl.reshape(B, 1))
        Vc_pp = torch.polar(V, th)
        iL = torch.tensor(self._lut).unsqueeze(0).expand(B, N)
        Vr = torch.zeros(B, self._nppc, dtype=x.dtype).scatter(1, iL, Vc_pp.real)
        Vi = torch.zeros(B, self._nppc, dtype=x.dtype).scatter(1, iL, Vc_pp.imag)
        Vc = torch.complex(Vr, Vi)
        Sb = Vc * torch.conj(Vc @ self._Ybus.T)
        Sf = Vc[:, torch.tensor(self._fb)] * torch.conj(Vc @ self._Yft.T)
        LUT = torch.tensor(self._lut)
        return torch.cat([V, -Sb.real[:, LUT], -Sb.imag[:, LUT], th, Sf.real, Sf.imag], dim=1)

    # ---- data conversion (physical shard units -> internal pu/rad) --------------------------
    def _z_of(self, node_x: np.ndarray, edge_x: np.ndarray) -> np.ndarray:
        b = self.baseMVA
        z = np.concatenate(
            [
                node_x[:, :, 0],
                node_x[:, :, 1] / b,
                node_x[:, :, 2] / b,
                np.deg2rad(node_x[:, :, 3]),
                edge_x[:, :, 0] / b,
                edge_x[:, :, 1] / b,
            ],
            axis=1,
        )
        return z[:, self.mask].astype(np.float64)

    def _truth_of(self, clean: np.ndarray) -> Dict[str, np.ndarray]:
        # clean [n,N,4] = [V, P, Q, theta] physical -> true state at keep buses + slack reference
        x = np.concatenate([np.deg2rad(clean[:, self.keep, 3]), clean[:, self.keep, 0]], axis=1)
        return {
            "x": x.astype(np.float64),
            "vsl": clean[:, self.slack, 0].astype(np.float64),
            "thsl": np.deg2rad(clean[:, self.slack, 3]).astype(np.float64),
        }

    # ---- fitting ----------------------------------------------------------------------------
    def fit(self, ds: "FdiaGraph", n_calib: int = 600) -> "SEBase":
        torch = _torch()
        self._build_network(ds)
        d = ds.to_numpy(["node_x", "edge_x", "family", "clean"])
        ben = np.where(d["family"] == 0)[0]
        if not len(ben):
            raise ValueError("fit needs benign records; pass the train split unfiltered")
        tr = self._truth_of(d["clean"][ben])
        self._fit_states(tr["x"])  # hook: subclasses learn their prior here
        self.xmean = tr["x"].mean(axis=0)
        # meter sigma = rms of benign residual AT THE TRUE STATE. The shard's meter error is a
        # constant bias plus jitter; a std across records cancels the bias and mis-weights, so the
        # total error about zero (the accuracy class) is the correct scale.
        c = ben[:n_calib]
        zc = self._z_of(d["node_x"][c], d["edge_x"][c])
        tc = self._truth_of(d["clean"][c])
        hz = self._h(tc["x"], tc["vsl"], tc["thsl"])
        self.sig = np.maximum(np.sqrt(((zc - hz) ** 2).mean(axis=0)), 1e-9)
        self.Wk = 1.0 / self.sig**2
        # chord Jacobian at the benign mean, weighted normal matrix, and its inverse
        from torch.func import jacrev, vmap

        def h1(xi, vi, ti):
            return self._h_t(xi[None], vi[None], ti[None])[0]

        x0 = torch.tensor(self.xmean, dtype=torch.float64)[None]
        v0 = torch.tensor([float(tr["vsl"][0])], dtype=torch.float64)
        t0 = torch.tensor([float(tr["thsl"][0])], dtype=torch.float64)
        self.H = vmap(jacrev(h1))(x0, v0, t0)[0].numpy()[self.mask]
        A = self.H.T @ (self.Wk[:, None] * self.H)
        self._Ai = self._inv(A)
        # residual covariance diagonal for normalized residuals; critical measurements excluded
        R = 1.0 / self.Wk
        om = R - np.einsum("ij,jk,ik->i", self.H, self._Ai, self.H)
        self.critical = om < 1e-6 * R
        self._om = np.maximum(om, 1e-12 * R)
        self._post_fit()
        return self

    def _fit_states(self, x_benign: np.ndarray) -> None:
        pass  # WLS learns nothing from the states; SubspacePrior overrides

    def _post_fit(self) -> None:
        pass  # hook for anything needing H/Wk (SubspacePrior builds its reduced system here)

    @staticmethod
    def _inv(A: np.ndarray) -> np.ndarray:
        # full-rank direct inverse; pinv only when genuinely singular (a cutoff on a full-rank
        # system silently amputates valid directions and pins the estimate near its start)
        ev = np.linalg.eigvalsh(0.5 * (A + A.T))
        if ev.min() > 0:
            return np.linalg.inv(A)
        return np.linalg.pinv(A, rcond=1e-12)

    # ---- solving ----------------------------------------------------------------------------
    def _basis(self) -> Optional[np.ndarray]:
        return None  # full state; SubspacePrior returns its VK

    def _solve_plain(self, z: np.ndarray, vsl: np.ndarray, thsl: np.ndarray) -> np.ndarray:
        """Batched chord-Newton with the shared weights (the WLS solve)."""
        VK = self._basis()
        B_, Ai = (
            (self.H, self._Ai)
            if VK is None
            else (self.H @ VK, self._inv((self.H @ VK).T @ (self.Wk[:, None] * (self.H @ VK))))
        )
        c = np.zeros((z.shape[0], B_.shape[1]))
        for _ in range(self.iters):
            x = self.xmean + (c @ VK.T if VK is not None else c)
            hz = self._h(x, vsl, thsl)
            c = c + ((z - hz) * self.Wk) @ B_ @ Ai.T
        return self.xmean + (c @ VK.T if VK is not None else c)

    def _w_solve(self, z: np.ndarray, w: np.ndarray, vsl: np.ndarray, thsl: np.ndarray) -> np.ndarray:
        """Chord-Newton with PER-RECORD weights and the divergence guard.

        The frozen Jacobian stops being a contraction when many measurements are down-weighted,
        so a step that raises the weighted residual (the SE objective) is rejected and the best
        iterate kept; non-finite iterates never propagate.
        """
        VK = self._basis()
        B_ = self.H if VK is None else self.H @ VK
        n, kd = z.shape[0], B_.shape[1]
        Ai = np.empty((n, kd, kd))
        for i in range(n):
            Ai[i] = self._inv(B_.T @ (w[i][:, None] * B_))
        c = np.zeros((n, kd))
        best_c, best_J = c.copy(), np.full(n, np.inf)
        for _ in range(self.iters):
            x = self.xmean + (c @ VK.T if VK is not None else c)
            hz = self._h(x, vsl, thsl)
            J = (w * (z - hz) ** 2).sum(axis=1)
            ok = np.isfinite(J) & (J < best_J)
            best_J = np.where(ok, J, best_J)
            best_c[ok] = c[ok]
            c = c + np.einsum("bij,bj->bi", Ai, ((z - hz) * w) @ B_)
            c = np.where(np.isfinite(c), c, best_c)
        return self.xmean + (best_c @ VK.T if VK is not None else best_c)

    def _nres(self, x: np.ndarray, z: np.ndarray, vsl: np.ndarray, thsl: np.ndarray) -> np.ndarray:
        """Residuals normalized by the residual covariance diagonal."""
        return np.abs(z - self._h(x, vsl, thsl)) / np.sqrt(self._om)[None, :]

    def _solve(self, z: np.ndarray, vsl: np.ndarray, thsl: np.ndarray) -> np.ndarray:
        return self._solve_plain(z, vsl, thsl)  # WLS; robust subclasses override

    # ---- public API -------------------------------------------------------------------------
    def estimate(self, ds: "FdiaGraph", chunk: int = 1000) -> np.ndarray:
        """Estimated states [n, 2(N-1)] = [theta rad | V pu] at non-slack buses, record order."""
        d = ds.to_numpy(["node_x", "edge_x", "clean"])
        tr = self._truth_of(d["clean"])  # slack reference only; the true state is never read here
        z = self._z_of(d["node_x"], d["edge_x"])
        out = np.empty((z.shape[0], self.SD))
        for s in range(0, z.shape[0], chunk):
            e = slice(s, s + chunk)
            out[e] = self._solve(z[e], tr["vsl"][e], tr["thsl"][e])
        return out

    def score(self, ds: "FdiaGraph", chunk: int = 1000) -> Dict[str, Dict[str, float]]:
        """Per-family angle (deg) and voltage (pu) MAE vs the clean truth, plus the geometric
        mean over families ('geo', the paper's table cell)."""
        from ..dataset import FAMILIES

        est = self.estimate(ds, chunk=chunk)
        d = ds.to_numpy(["family", "clean"])
        tr = self._truth_of(d["clean"])
        ns = len(self.keep)
        err = est - tr["x"]
        ang = np.abs(err[:, :ns]).mean(axis=1) * 180.0 / np.pi
        volt = np.abs(err[:, ns:]).mean(axis=1)
        out: Dict[str, Dict[str, float]] = {}
        geo_a, geo_v = [], []
        for fid, name in FAMILIES.items():
            m = d["family"] == fid
            if not m.any():
                continue
            out[name] = {"angle_mae_deg": float(ang[m].mean()), "voltage_mae_pu": float(volt[m].mean())}
            geo_a.append(ang[m].mean())
            geo_v.append(volt[m].mean())
        out["geo"] = {
            "angle_mae_deg": float(np.exp(np.mean(np.log(geo_a)))),
            "voltage_mae_pu": float(np.exp(np.mean(np.log(geo_v)))),
        }
        return out
