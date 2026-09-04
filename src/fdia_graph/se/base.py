"""State estimation on fdia-graph shards — the shared machinery behind every method class.

SEBase owns what all estimators have in common: the AC measurement model h(x) built from the
pandapower case, the chord-Newton iteration with its divergence guard, the meter weights calibrated
from benign residuals, and the reference handling (the classical 2N-1 state: only the slack ANGLE
is fixed, pinned per record to the clean layer so estimates and truth share one frame; every
voltage magnitude including the slack is estimated, matching production practice). Subclasses change only the state space and the weights, mirroring the paper's protocol.

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


def _scipy_linalg():
    try:
        import scipy.linalg

        return scipy.linalg
    except ImportError as e:
        raise ImportError("state estimation needs scipy: pip install 'fdia-graph[se]'") from e


class SEBase:
    """Weighted least squares AC state estimation, the audited baseline of the paper.

    Usage:
        est = WLS().fit(fg.load("ieee118", split="train"))
        xhat = est.estimate(test_ds)      # [n, 2N-1] = [theta rad (non-slack) | V pu (all buses)]
        rep  = est.score(test_ds)         # per-family angle/voltage MAE vs the clean truth

    fit() calibrates per-meter error scales as the rms of benign residuals at the true state
    (the accuracy-class total error, bias included) and freezes the Jacobian at the benign mean
    state, the chord that makes per-record solves cheap.
    """

    def __init__(self, npass: int = 40, iters: int = 8) -> None:
        if npass < 1 or iters < 1:
            raise ValueError(f"npass and iters must be >= 1, got {npass}, {iters}")
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
        self.keep = np.array([i for i in range(self.N) if i != self.slack])  # angle buses
        # Classical 2N-1 state: angles at every non-slack bus, voltage magnitude at EVERY bus.
        # Only the slack angle is fixed (the reference the math requires); the slack voltage is
        # estimated like any other, matching production practice and pandapower's estimator.
        self.SD = len(self.keep) + self.N
        # measurement mask, constant across records: [V(N), P(N), Q(N), theta(N), Pf(E), Qf(E)]
        nm = ds[0]["node_m"].numpy().astype(bool)
        em = ds[0]["edge_m"].numpy().astype(bool)
        self.mask = np.concatenate([nm[:, 0], nm[:, 1], nm[:, 2], nm[:, 3], em[:, 0], em[:, 1]])
        self.m = int(self.mask.sum())

    def _h(self, x: np.ndarray, thsl: np.ndarray) -> np.ndarray:
        """Masked measurement prediction for a batch of states (slack angle pinned per record)."""
        torch = _torch()
        with torch.no_grad():
            return self._h_t(
                torch.tensor(x, dtype=torch.float64),
                torch.tensor(thsl, dtype=torch.float64),
            ).numpy()[:, self.mask]

    def _h_t(self, x: Any, thsl: Any) -> Any:
        # Full AC forward map (torch, unmasked). Shard injections are load-positive, so bus
        # injections are emitted as -S; flows are from-end. Exactly the paper's h.
        torch = _torch()
        B, N, ns = x.shape[0], self.N, len(self.keep)
        thsl = thsl.to(x.dtype)
        idx = torch.tensor(self.keep).unsqueeze(0).expand(B, ns)
        sidx = torch.full((B, 1), self.slack, dtype=torch.long)
        th = torch.zeros(B, N, dtype=x.dtype).scatter(1, idx, x[:, :ns]).scatter(1, sidx, thsl.reshape(B, 1))
        V = x[:, ns:]  # voltage magnitude at every bus is part of the state
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
        # clean [n,N,4] = [V, P, Q, theta] physical -> true 2N-1 state + slack angle reference
        x = np.concatenate([np.deg2rad(clean[:, self.keep, 3]), clean[:, :, 0]], axis=1)
        return {
            "x": x.astype(np.float64),
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
        hz = self._h(tc["x"], tc["thsl"])
        self.sig = np.maximum(np.sqrt(((zc - hz) ** 2).mean(axis=0)), 1e-9)
        self.Wk = 1.0 / self.sig**2
        # chord Jacobian at the benign mean, weighted normal matrix, and its inverse
        from torch.func import jacrev, vmap

        def h1(xi, ti):
            return self._h_t(xi[None], ti[None])[0]

        x0 = torch.tensor(self.xmean, dtype=torch.float64)[None]
        t0 = torch.tensor([float(tr["thsl"][0])], dtype=torch.float64)
        self.H = vmap(jacrev(h1))(x0, t0)[0].numpy()[self.mask]
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
        # Direct inverse of a symmetric positive-definite normal matrix, pinv only when it is
        # (numerically) singular. Positive-definiteness is decided by a Cholesky factorization and
        # near-singularity by LAPACK's condition estimate of the triangular factor, both O(n^2) after
        # the factorization; an eigen-decomposition here cost 240 ms per call at IEEE-300 size and
        # was the reason every robust arm (one inverse per record per pass) took hours to days.
        lapack = _scipy_linalg().lapack

        A = np.asarray(A)
        eps = np.finfo(A.dtype if np.issubdtype(A.dtype, np.floating) else np.float64).eps
        S = 0.5 * (A + A.T)
        try:
            L = np.linalg.cholesky(S)
            rcond, _ = lapack.dtrcon(L, norm="1", uplo="L", diag="N")
            if rcond**2 > 100 * eps:  # cond(A) ~ cond(L)^2; the same 1e-14 relative floor as before
                return np.linalg.inv(S)
        except np.linalg.LinAlgError:
            pass
        return np.linalg.pinv(S, rcond=100 * eps)

    @staticmethod
    def _normal_matrices(w: np.ndarray, B_: np.ndarray, sub: int = 50) -> np.ndarray:
        """Per-record weighted normal matrices B^T diag(w_i) B as [n, k, k], built in sub-batches so
        the [sub, k, m] intermediate stays small. One einsum over the whole chunk materialized an
        8 GB intermediate at IEEE-300 size and took 260 s per 200 records; this takes 1.4 s."""
        if sub < 1:
            raise ValueError(f"sub must be a positive sub-batch size, got {sub}")
        n, k = w.shape[0], B_.shape[1]
        out = np.empty((n, k, k), dtype=np.result_type(w, B_))
        BT = B_.T[None]  # [1, k, m]
        for a in range(0, n, sub):
            out[a : a + sub] = (BT * w[a : a + sub, None, :]) @ B_
        return out

    @classmethod
    def _inv_batch(cls, A: np.ndarray) -> np.ndarray:
        """Inverses of a stack of normal matrices [n, k, k] through torch's batched Cholesky (about
        100x faster than NumPy's batched inverse on this LAPACK build: 0.2 s per 1000 records at
        IEEE-118 size, 1 s at IEEE-300), falling back to the guarded per-matrix path only for the
        members that are not positive definite. The per-record Python loop this replaces, and the
        NumPy batched inverse after it, were the dominant cost of every robust arm."""
        torch = _torch()
        S = 0.5 * (A + np.swapaxes(A, 1, 2))
        L, info = torch.linalg.cholesky_ex(torch.from_numpy(S))
        good = info.numpy() == 0
        out = np.empty_like(S)
        if good.any():
            out[good] = torch.cholesky_inverse(L[torch.from_numpy(good)]).numpy()
        for i in np.where(~good)[0]:  # not positive definite (e.g. a removal set): guarded path
            out[i] = cls._inv(S[i])
        return out

    @staticmethod
    def _cond(A: np.ndarray, its: int = 40) -> float:
        """Spectral condition number of a symmetric positive-definite matrix, inf if it is not PD.

        Cholesky for the PD test, then power iteration for the largest eigenvalue and inverse
        iteration through the Cholesky factor for the smallest; tens of milliseconds where a full
        eigen-decomposition takes hundreds, and equal to it to three decimals on the real normal
        matrices (checked on IEEE 14/118 with and without removed meters)."""
        sl = _scipy_linalg()
        cho_factor, cho_solve = sl.cho_factor, sl.cho_solve

        S = 0.5 * (A + A.T)
        try:
            cf = cho_factor(S, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            return float("inf")
        n = S.shape[0]
        v = np.ones(n) / np.sqrt(n)
        for _ in range(its):
            v = S @ v
            v /= np.linalg.norm(v)
        lmax = float(v @ (S @ v))
        u = np.ones(n) / np.sqrt(n)
        for _ in range(its):
            u = cho_solve(cf, u, check_finite=False)
            u /= np.linalg.norm(u)
        lmin = float(u @ (S @ u))
        return lmax / max(lmin, 1e-300)

    # ---- solving ----------------------------------------------------------------------------
    def _basis(self) -> Optional[np.ndarray]:
        return None  # full state; SubspacePrior returns its VK

    def _solve_plain(self, z: np.ndarray, thsl: np.ndarray) -> np.ndarray:
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
            hz = self._h(x, thsl)
            c = c + ((z - hz) * self.Wk) @ B_ @ Ai.T
        return self.xmean + (c @ VK.T if VK is not None else c)

    def _w_solve(self, z: np.ndarray, w: np.ndarray, thsl: np.ndarray) -> np.ndarray:
        """Chord-Newton with PER-RECORD weights and the divergence guard.

        The frozen Jacobian stops being a contraction when many measurements are down-weighted,
        so a step that raises the weighted residual (the SE objective) is rejected and the best
        iterate kept; non-finite iterates never propagate.
        """
        VK = self._basis()
        B_ = self.H if VK is None else self.H @ VK
        n, kd = z.shape[0], B_.shape[1]
        Ai = self._inv_batch(self._normal_matrices(w, B_))
        c = np.zeros((n, kd))
        best_c, best_J = c.copy(), np.full(n, np.inf)
        for _ in range(self.iters):
            x = self.xmean + (c @ VK.T if VK is not None else c)
            hz = self._h(x, thsl)
            J = (w * (z - hz) ** 2).sum(axis=1)
            ok = np.isfinite(J) & (J < best_J)
            best_J = np.where(ok, J, best_J)
            best_c[ok] = c[ok]
            c = c + np.einsum("bij,bj->bi", Ai, ((z - hz) * w) @ B_)
            c = np.where(np.isfinite(c), c, best_c)
        return self.xmean + (best_c @ VK.T if VK is not None else best_c)

    def _nres(self, x: np.ndarray, z: np.ndarray, thsl: np.ndarray) -> np.ndarray:
        """Residuals normalized by the residual covariance diagonal."""
        return np.abs(z - self._h(x, thsl)) / np.sqrt(self._om)[None, :]

    def _solve(self, z: np.ndarray, thsl: np.ndarray) -> np.ndarray:
        return self._solve_plain(z, thsl)  # WLS; robust subclasses override

    # ---- public API -------------------------------------------------------------------------
    def estimate(self, ds: "FdiaGraph", chunk: int = 1000) -> np.ndarray:
        """Estimated states [n, 2N-1] = [theta rad (non-slack) | V pu (all buses)], record order."""
        d = ds.to_numpy(["node_x", "edge_x", "clean"])
        tr = self._truth_of(d["clean"])  # slack angle reference only; the true state is never read here
        z = self._z_of(d["node_x"], d["edge_x"])
        out = np.empty((z.shape[0], self.SD))
        for s in range(0, z.shape[0], chunk):
            e = slice(s, s + chunk)
            out[e] = self._solve(z[e], tr["thsl"][e])
        return out

    def score(
        self, ds: "FdiaGraph", chunk: int = 1000, xhat: Optional[np.ndarray] = None
    ) -> Dict[str, Dict[str, float]]:
        """Per-family angle (deg) and voltage (pu) MAE vs the clean truth, plus the geometric
        mean over families ('geo', the paper's table cell). Pass `xhat` (a previous `estimate(ds)`)
        to score without re-solving, e.g. from a cache; it must be in record order of `ds`."""
        from ..dataset import FAMILIES

        est = self.estimate(ds, chunk=chunk) if xhat is None else np.asarray(xhat, np.float64)
        if est.shape != (len(ds), self.SD):
            raise ValueError(f"xhat must be [{len(ds)}, {self.SD}], got {est.shape}")
        d = ds.to_numpy(["family", "clean"])
        tr = self._truth_of(d["clean"])
        ns = len(self.keep)  # angle block; voltage block covers ALL N buses (2N-1 state)
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
            geo_a.append(max(float(ang[m].mean()), 1e-30))  # guard log(0) on degenerate slices
            geo_v.append(max(float(volt[m].mean()), 1e-30))
        out["geo"] = {
            "angle_mae_deg": float(np.exp(np.mean(np.log(geo_a)))),
            "voltage_mae_pu": float(np.exp(np.mean(np.log(geo_v)))),
        }
        return out
