"""State estimation on fdia-graph datasets — the shared machinery behind every method class.

SEBase owns what all estimators have in common: the AC measurement model h(x) built from the
pandapower case, the chord-Newton iteration with its divergence guard, the meter weights calibrated
from benign residuals, and the reference handling (the classical 2N-1 state: only the slack ANGLE
is fixed, to one reference angle `theta_ref` taken from the training split at fit time (the
case's reference angle on the released pools); every voltage magnitude including the slack is
estimated, matching production practice). Estimating reads measurements only; fitting calibrates on the training
split's clean layer, and `score` compares with it. Subclasses change only the state space and the
weights, mirroring the paper's protocol.

Needs pandapower and scipy: pip install "fdia-graph[se]". The measurement function and its Jacobian
are the closed-form numpy kernel (`formulas.network.ac_measurement`, `ac_jacobian`); torch, when
installed, only speeds up the per-record inverses. Datasets must carry the clean layer (a timeline,
or a v0.7.2 record shard; it supplies the truth) loaded with units="physical" (the default).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from ..formulas.estimation import (
    accuracy_class_sigma,
    critical_measurements,
    floored_covariance,
    huber_weights,
    normal_matrix,
    normalized_residual,
    residual_covariance_diag,
    weighted_objective,
    wls_step,
    wls_step_batched,
)
from ..formulas.linalg import batched_normal_matrices, condition_number, guarded_inverse
from ..formulas.network import ac_jacobian, ac_measurement
from ..models.data import TrueState  # noqa: F401  re-exported: defined here before the models package
from ..models.grid import EDGE, NODE, EdgeColumns, NodeColumns
from ..models.scores import (  # noqa: F401  re-exported: defined here before the models package
    ErrorPair,
    EstimatorScores,
)

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
        raise ImportError("the differentiable twin _h_t needs torch: pip install 'fdia-graph[torch]'") from e


def require_physical(ds: FdiaGraph) -> None:
    """Refuse a units="pu" view: the estimators convert the stored physical units themselves, so a
    per-unit view would be converted twice and every estimate silently off by baseMVA."""
    if ds.units != "physical":
        raise ValueError("state estimation expects units='physical' datasets (the default)")


def _torch_or_none():
    try:
        import torch

        return torch
    except ImportError:
        return None


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
    def _build_network(self, ds: FdiaGraph, need_clean: bool = True) -> None:
        try:
            import pandapower as pp
            import pandapower.networks as pn
            from pandapower.pypower.makeYbus import makeYbus
        except ImportError as e:
            raise ImportError("state estimation needs pandapower: pip install 'fdia-graph[se]'") from e
        require_physical(ds)
        if need_clean and not ds.has_clean:
            raise ValueError(
                "dataset has no clean layer; load a timeline or a v0.7.2 record shard, "
                'or fit with calibrate="measured"'
            )
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
        self._Ybus_np = np.asarray(Yb.todense())
        self._Yf_np = np.asarray(Yf.todense())
        torch = _torch_or_none()
        if torch is not None:  # the torch twin of h, kept for callers that differentiate through it
            self._Ybus = torch.tensor(self._Ybus_np, dtype=torch.complex128)
            self._Yft = torch.tensor(self._Yf_np, dtype=torch.complex128)
        self.slack = int(net.ext_grid.bus.values[0])
        # the case's reference angle (rad), until fit() takes the training split's (the same on the
        # released pools); either way one constant, so no record's true state fixes an estimate's frame
        self.theta_ref = float(np.deg2rad(net.ext_grid.va_degree.values[0]))
        if ds.slack is not None and int(ds.slack) != self.slack:  # both index buses 0..N-1 on every case
            raise ValueError(f"the case's slack is bus {self.slack}, the dataset's is {int(ds.slack)}")
        self.keep = np.array([i for i in range(self.N) if i != self.slack])  # angle buses
        # the case's own power-flow solution, the starting point of a measurement-only calibration
        res = net.res_bus.reindex(sorted(net.bus.index))
        self._x_case = np.concatenate(
            [np.deg2rad(res["va_degree"].to_numpy()[self.keep]), res["vm_pu"].to_numpy()]
        ).astype(np.float64)
        # Classical 2N-1 state: angles at every non-slack bus, voltage magnitude at EVERY bus.
        # Only the slack angle is fixed (the reference the math requires); the slack voltage is
        # estimated like any other, matching production practice and pandapower's estimator.
        self.SD = len(self.keep) + self.N
        # measurement mask, constant across records: [V(N), P(N), Q(N), theta(N), Pf(E), Qf(E)]
        masks = ds.export(["node_m", "edge_m"])  # numpy, so the estimator does not need torch
        nm = masks["node_m"][0].astype(bool)
        em = masks["edge_m"][0].astype(bool)
        self.mask = np.concatenate([*NodeColumns.of(nm), *EdgeColumns.of(em)])
        self.m = int(self.mask.sum())

    def _angles(self, x: np.ndarray, thsl: np.ndarray) -> np.ndarray:
        """Every bus angle [n, N]: the state's non-slack angles and the pinned slack angle."""
        th = np.zeros((x.shape[0], self.N))
        th[:, self.keep] = x[:, : len(self.keep)]
        th[:, self.slack] = thsl
        return th

    def _h(self, x: np.ndarray, thsl: np.ndarray) -> np.ndarray:
        """Masked measurement prediction for a batch of states (slack angle pinned per record):
        `formulas.network.ac_measurement` at the state's angles and magnitudes."""
        ns = len(self.keep)
        full = ac_measurement(
            x[:, ns:], self._angles(x, thsl), self._Ybus_np, self._Yf_np, self._fb, self._lut, self._nppc
        )
        return full[:, self.mask]

    def _h_ref(self, x: np.ndarray) -> np.ndarray:
        """`_h` at the fitted reference angle (`ref_angles`), the prediction every solve and residual
        uses; `_h` itself takes the slack angle per record for the truth calibration."""
        return self._h(x, self.ref_angles(len(x)))

    def _jacobian(self, x: np.ndarray, thsl: float) -> np.ndarray:
        """The masked measurement Jacobian [m, SD] at one state, in closed form
        (`formulas.network.ac_jacobian`); the slack angle column is dropped."""
        ns, N = len(self.keep), self.N
        th = self._angles(x[None], np.array([thsl]))[0]
        J = ac_jacobian(x[ns:], th, self._Ybus_np, self._Yf_np, self._fb, self._lut, self._nppc)
        cols = np.concatenate([self.keep, N + np.arange(N)])
        return J[:, cols][self.mask]

    def _h_t(self, x: Any, thsl: Any) -> Any:
        """The torch twin of `_h`, unmasked: the same AC forward map, differentiable, kept for
        callers that take gradients through it (the estimator itself uses the numpy kernel and the
        closed-form Jacobian). Needs torch."""
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

    # ---- data conversion (physical dataset units -> internal pu/rad) --------------------------
    def _z_of(self, node_x: np.ndarray, edge_x: np.ndarray) -> np.ndarray:
        b = self.baseMVA
        z = np.concatenate(
            [
                node_x[:, :, NODE.v],
                node_x[:, :, NODE.p_inj] / b,
                node_x[:, :, NODE.q_inj] / b,
                np.deg2rad(node_x[:, :, NODE.theta]),
                edge_x[:, :, EDGE.p_from] / b,
                edge_x[:, :, EDGE.q_from] / b,
            ],
            axis=1,
        )
        return z[:, self.mask].astype(np.float64)

    def _truth_of(self, clean: np.ndarray) -> TrueState:
        # clean [n,N,4] = [V, P, Q, theta] physical -> true 2N-1 state + slack angle reference
        x = np.concatenate([np.deg2rad(clean[:, self.keep, NODE.theta]), clean[:, :, NODE.v]], axis=1)
        return TrueState(
            x.astype(np.float64), np.deg2rad(clean[:, self.slack, NODE.theta]).astype(np.float64)
        )

    # ---- fitting ----------------------------------------------------------------------------
    def fit(self, ds: FdiaGraph, n_calib: int = 600, calibrate: str = "truth") -> SEBase:
        """Calibrate on the benign records of `ds`. calibrate="truth" (the estimation benchmark)
        takes the meter errors, the benign mean state and the angle reference from the training
        split's clean layer; calibrate="measured" takes them from equipment data and measurements
        alone (`_fit_from_measurements`), for any path whose output feeds a detector."""
        if calibrate not in ("truth", "measured"):
            raise ValueError(f"calibrate must be 'truth' or 'measured', got {calibrate!r}")
        self._build_network(ds, need_clean=calibrate == "truth")
        d = ds.export(["node_x", "edge_x", "family"] + (["clean"] if calibrate == "truth" else []))
        ben = np.where(d["family"] == 0)[0]
        if not len(ben):
            raise ValueError("fit needs benign records; pass the train split unfiltered")
        # The calibration records are spread evenly over the benign set: on a timeline the first ones
        # are one early stretch of the year, a single load regime, and the meter error scales with the
        # reading.
        c = ben[np.linspace(0, len(ben) - 1, min(n_calib, len(ben))).round().astype(int)]
        zc = self._z_of(d["node_x"][c], d["edge_x"][c])
        if calibrate == "truth":
            self._fit_from_truth(d["clean"][ben], d["clean"][c], zc)
        else:
            self._fit_from_measurements(self._z_of(d["node_x"][ben], d["edge_x"][ben]), zc)
        self.calibrate = calibrate
        self._post_fit()
        return self

    def _fit_from_truth(self, clean_ben: np.ndarray, clean_c: np.ndarray, zc: np.ndarray) -> None:
        """Meter sigma = rms of the benign residual AT THE TRUE STATE. The dataset's meter error is a
        constant bias plus jitter; a std across records cancels the bias and mis-weights, so the total
        error about zero (the accuracy class) is the correct scale."""
        tr = self._truth_of(clean_ben)
        self._fit_reference(tr["thsl"])
        self._fit_states(tr["x"])  # hook: subclasses learn their prior here
        tc = self._truth_of(clean_c)
        sig = np.sqrt(((zc - self._h(tc["x"], tc["thsl"])) ** 2).mean(axis=0))
        self._set_model(tr["x"].mean(axis=0), sig)

    def _fit_from_measurements(self, z_ben: np.ndarray, zc: np.ndarray, passes: int = 2) -> None:
        """The calibration from equipment data and measurements alone. Every meter's sigma is its
        accuracy class (`_class_sigma`): a meter's constant bias cannot be told apart from the state by
        measurements, so residual-based sigmas shrink on the biased meters and collapse the fit. The
        angle reference is the case's; the linearization point starts at the case's power-flow
        solution and moves to the mean estimate of the calibration scans each pass. The subclass
        prior is learned from the estimated benign states."""
        sig, x = self._class_sigma(np.abs(zc).mean(axis=0)), self._x_case
        for _ in range(passes):
            self._set_model(x, sig, full_state=True)
            x = self._solve_plain(zc).mean(axis=0)
        self._set_model(x, sig, full_state=True)
        self._fit_states(self._solve_plain(z_ben))
        self._set_model(x, sig)

    def _class_sigma(self, mean_abs: np.ndarray) -> np.ndarray:
        """The accuracy-class sigma of every metered slot [m], in the estimator's units (pu, rad)."""
        from ..engine.base import ACCURACY_CLASS, POWER_NOISE_FLOOR_MW

        N, E = self.N, self.E
        order = [
            ("v", N, False),
            ("pi", N, True),
            ("qi", N, True),
            ("va", N, False),
            ("pf", E, True),
            ("qf", E, True),
        ]
        cls = np.concatenate([np.full(n, ACCURACY_CLASS[k]) for k, n, _ in order])[self.mask]
        rel = np.concatenate([np.full(n, r) for _, n, r in order])[self.mask]
        return accuracy_class_sigma(mean_abs, cls, rel, POWER_NOISE_FLOOR_MW / self.baseMVA)

    def _set_model(self, xmean: np.ndarray, sig: np.ndarray, full_state: bool = False) -> None:
        """The linearized model at `xmean` with meter errors `sig`: the weights, the chord Jacobian, the
        inverse normal matrix, the residual covariance with the critical meters, and the solve's
        Jacobian in the state or the subclass subspace (`full_state` forces the full state, for a
        calibration pass before the subspace is learned)."""
        self._full_state = full_state
        self.xmean = xmean
        self.sig = np.maximum(sig, 1e-9)
        self.Wk = 1.0 / self.sig**2
        self.H = self._jacobian(self.xmean, self.theta_ref)
        self._Ai = guarded_inverse(normal_matrix(self.H, self.Wk))
        om, R = residual_covariance_diag(self.H, self.Wk, self._Ai)
        self.critical = critical_measurements(om, R)
        self._om = floored_covariance(om, R)
        VK = self._basis()
        self._B, self._Bi = (
            (self.H, self._Ai)
            if VK is None
            else (self.H @ VK, guarded_inverse(normal_matrix(self.H @ VK, self.Wk)))
        )

    def _fit_states(self, x_benign: np.ndarray) -> None:
        pass  # WLS learns nothing from the states; SubspacePrior overrides

    def _post_fit(self) -> None:
        pass  # hook for anything needing H/Wk (SubspacePrior builds its reduced system here)

    @property
    def is_fitted(self) -> bool:
        """True once fit() has run (the chord Jacobian exists)."""
        return hasattr(self, "H")

    @staticmethod
    def _inv(A: np.ndarray) -> np.ndarray:
        """Guarded inverse of a normal matrix: `formulas.linalg.guarded_inverse`."""
        return guarded_inverse(A)

    @staticmethod
    def _normal_matrices(w: np.ndarray, B_: np.ndarray, sub: int = 50) -> np.ndarray:
        """Per-record normal matrices: `formulas.linalg.batched_normal_matrices`."""
        return batched_normal_matrices(w, B_, sub)

    @classmethod
    def _inv_batch(cls, A: np.ndarray) -> np.ndarray:
        """Inverses of a stack of normal matrices [n, k, k] through torch's batched Cholesky when
        torch is installed (about 100x faster than NumPy's batched inverse on this LAPACK build:
        0.2 s per 1000 records at IEEE-118 size, 1 s at IEEE-300), falling back to the guarded
        per-matrix path for the members that are not positive definite, or for every member
        without torch. The per-record Python loop this replaces, and the NumPy batched inverse
        after it, were the dominant cost of every robust arm."""
        S = 0.5 * (A + np.swapaxes(A, 1, 2))
        torch = _torch_or_none()
        if torch is None:
            return np.stack([cls._inv(Si) for Si in S])
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
        """Spectral condition number of a normal matrix: `formulas.linalg.condition_number`."""
        return condition_number(A, its)

    # ---- solving ----------------------------------------------------------------------------
    def _basis(self) -> Optional[np.ndarray]:
        """The basis the solve runs in: the subclass subspace, or None (the full state) for a model
        built with full_state=True, a calibration pass before the subspace is learned."""
        return None if getattr(self, "_full_state", False) else self._subspace()

    def _subspace(self) -> Optional[np.ndarray]:
        return None  # full state; SubspacePrior returns its VK

    def _solve_plain(self, z: np.ndarray) -> np.ndarray:
        """Batched chord-Newton with the shared weights (the WLS solve)."""
        VK = self._basis()
        B_, Ai = self._B, self._Bi  # built once in fit
        c = np.zeros((z.shape[0], B_.shape[1]))
        for _ in range(self.iters):
            x = self.xmean + (c @ VK.T if VK is not None else c)
            hz = self._h_ref(x)
            c = c + wls_step(z - hz, self.Wk, B_, Ai)
        return self.xmean + (c @ VK.T if VK is not None else c)

    def _w_solve(self, z: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Chord-Newton with PER-RECORD weights and the divergence guard.

        The frozen Jacobian stops being a contraction when many measurements are down-weighted,
        so a step that raises the weighted residual (the SE objective) is rejected and the best
        iterate kept; non-finite iterates never propagate.
        """
        VK = self._basis()
        B_ = self._B
        n, kd = z.shape[0], B_.shape[1]
        Ai = self._inv_batch(batched_normal_matrices(w, B_))
        c = np.zeros((n, kd))
        best_c, best_J = c.copy(), np.full(n, np.inf)
        for _ in range(self.iters):
            x = self.xmean + (c @ VK.T if VK is not None else c)
            hz = self._h_ref(x)
            J = weighted_objective(z - hz, w)
            ok = np.isfinite(J) & (J < best_J)
            best_J = np.where(ok, J, best_J)
            best_c[ok] = c[ok]
            c = c + wls_step_batched(z - hz, w, B_, Ai)
            c = np.where(np.isfinite(c), c, best_c)
        # the last step is a candidate too, so a weighted solve takes as many steps as the plain one
        J = weighted_objective(z - self._h_ref(self.xmean + (c @ VK.T if VK is not None else c)), w)
        ok = np.isfinite(J) & (J < best_J)
        best_c[ok] = c[ok]
        return self.xmean + (best_c @ VK.T if VK is not None else best_c)

    def _nres(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Residuals normalized by the residual covariance diagonal [HAN75]."""
        return normalized_residual(z - self._h_ref(x), self._om)

    def _huber_passes(self, x: np.ndarray, z: np.ndarray, w: np.ndarray, c: float, tol: float) -> np.ndarray:
        """Huber reweighting on the estimate's own residual [HUB64]: a_i = min(1, c / |r_N,i|), re-solve
        with w * a, until no weight moves by more than tol or npass passes are done."""
        prev = None
        for _ in range(self.npass):
            a = huber_weights(self._nres(x, z), c)
            if prev is not None and np.abs(a - prev).max() < tol:
                break  # weights settled: further passes reproduce the same estimate
            x = self._w_solve(z, w * a)
            prev = a
        return x

    def _solve(self, z: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
        """One chunk of records. w: per-record weights [n, m] from `_record_weights`, None for Wk."""
        return self._solve_plain(z) if w is None else self._w_solve(z, w)

    def _record_weights(self, ds: FdiaGraph) -> Optional[np.ndarray]:
        """Per-record meter weights [n, m] the method derives from the dataset itself (a gate, a
        temporal residual), or None for the shared Wk. Every path that solves a dataset goes
        through here, so a composed estimator (a localizer's, a trust selector's) is the same
        estimator `estimate` runs."""
        return None

    def _estimate_arrays(self, z: np.ndarray, w: Optional[np.ndarray], chunk: int = 1000) -> np.ndarray:
        """Solve converted measurements [n, m] in chunks, with optional per-record weights."""
        out = np.empty((z.shape[0], self.SD))
        for s in range(0, z.shape[0], chunk):
            e = slice(s, s + chunk)
            out[e] = self._solve(z[e], None if w is None else w[e])
        return out

    # ---- public API -------------------------------------------------------------------------
    def estimate(self, ds: FdiaGraph, chunk: int = 1000) -> np.ndarray:
        """Estimated states [n, 2N-1] = [theta rad (non-slack) | V pu (all buses)], record order."""
        require_physical(ds)
        d = ds.export(["node_x", "edge_x"])
        z = self._z_of(d["node_x"], d["edge_x"])
        return self._estimate_arrays(z, self._record_weights(ds), chunk)

    def _fit_reference(self, thsl: np.ndarray) -> None:
        """The angle reference from the training truth, part of the fit's calibration: the slack
        angle must be one constant over the split (the case's `va_degree` for the released pools, a
        custom pool may use another), and every estimate afterwards uses it without reading truth."""
        if not np.allclose(thsl, thsl[0], atol=1e-9):
            raise ValueError(
                "the slack angle varies across the training frames; estimates need one fixed reference"
            )
        self.theta_ref = float(thsl[0])

    def ref_angles(self, n: int) -> np.ndarray:
        """The slack angle every estimate is referenced to, for n records: `theta_ref`, fixed at fit time."""
        return np.full(n, self.theta_ref)

    def score(self, ds: FdiaGraph, chunk: int = 1000, xhat: Optional[np.ndarray] = None) -> EstimatorScores:
        """Per-family angle (deg) and voltage (pu) MAE vs the clean truth, plus the geometric
        mean over families ('geo', the paper's table cell). Pass `xhat` (a previous `estimate(ds)`)
        to score without re-solving, e.g. from a cache; it must be in record order of `ds`."""
        from ..dataset import FAMILIES

        require_physical(ds)
        est = self.estimate(ds, chunk=chunk) if xhat is None else np.asarray(xhat, np.float64)
        if est.shape != (len(ds), self.SD):
            raise ValueError(f"xhat must be [{len(ds)}, {self.SD}], got {est.shape}")
        d = ds.export(["family", "clean"])
        tr = self._truth_of(d["clean"])
        ns = len(self.keep)  # angle block; voltage block covers ALL N buses (2N-1 state)
        err = est - tr["x"]
        ang = np.abs(err[:, :ns]).mean(axis=1) * 180.0 / np.pi
        volt = np.abs(err[:, ns:]).mean(axis=1)
        out: dict[str, dict[str, float]] = {}
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
        return EstimatorScores(**{k: ErrorPair(**v) for k, v in out.items()})
