"""Direct tests for public pieces the rest of the suite reached only through a whole pipeline: the
replay policy, the linear-algebra and network kernels on cases small enough to check by hand, the
episode reader, the release lookup, the trust selector base, the removal guard and the learned
localizer's threshold."""

import h5py
import numpy as np


def test_local_ac_solve_recovers_a_two_bus_voltage():
    from fdia_graph.formulas.network import local_ac_solve

    y = 1.0 / (0.01 + 0.1j)
    Y = np.array([[y, -y], [-y, y]])
    V_true = np.array([1.0 + 0j, 0.97 * np.exp(-0.05j)])  # bus 0 held, bus 1 the interior
    S1 = V_true[1] * np.conj(Y[1] @ V_true)
    V = local_ac_solve(Y, np.array([1.0 + 0j, 1.0 + 0j]), np.array([1]), np.array([S1]))
    assert V is not None and abs(V[1] - V_true[1]) < 1e-8 and V[0] == 1.0


# ---- sequences and episodes ----------------------------------------------------------------------------


def test_per_bus_sequences_regroup_by_bus():
    from fdia_graph.dataset.sequence import per_bus_sequences

    Xw = np.arange(2 * 3 * 4 * 1, dtype=float).reshape(2, 3, 4, 1)  # [n, W, N, C]
    yf = (Xw[..., 0] > 10).astype(np.uint8)  # [n, W, N]
    X, y = per_bus_sequences(Xw, yf, "frame")
    assert X.shape == (8, 3, 1) and y.shape == (8, 3)
    assert np.array_equal(X[5, :, 0], Xw[1, :, 1, 0])  # record 1, bus 1
    assert np.array_equal(y[5], yf[1, :, 1])  # its per-frame labels follow it
    assert np.array_equal(y, yf.transpose(0, 2, 1).reshape(8, 3))
    ya = yf.any(axis=1)  # [n, N] window labels
    X, y = per_bus_sequences(Xw, ya, "any")
    assert y.shape == (8,) and np.array_equal(y, ya.reshape(8)) and y[5] == ya[1, 1]


def test_read_episodes_reads_the_group(tmp_path):
    from fdia_graph import schema
    from fdia_graph.dataset.sequence import read_episodes

    path = tmp_path / "e.h5"
    with h5py.File(path, "w") as f:
        g = f.create_group(schema.Group.EPISODES)
        g[schema.EPISODE_ONSET] = [3, 10]
        g[schema.EPISODE_LENGTH] = [2, 1]
        g[schema.EPISODE_FAMILY] = [1, 4]
        g[schema.EPISODE_BUS_PTR] = [0, 2, 3]
        g[schema.EPISODE_BUS_IDX] = [5, 7, 2]
    with h5py.File(path, "r") as f:
        e = read_episodes(f, True)
        assert read_episodes(f, False) is None
    assert e.onset.tolist() == [3, 10] and e.family.tolist() == [1, 4]
    assert [b.tolist() for b in e.buses] == [[5, 7], [2]]
