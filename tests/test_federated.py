"""fdia_graph.federated: the partition of buses into clients and the averaging formulas.

The parity case is the federated localization paper's own IEEE-14 partitions (spectral clustering,
random_state 42, as cached with its runs), reproduced from the test timeline's topology."""

import numpy as np
import pytest

from fdia_graph.federated import compute_nodes, partition_from_assignment, spectral_partition
from fdia_graph.formulas.federated import attackable_affinity, fedavg, halo_nodes

# The paper's cached IEEE-14 partitions: assignment, cut edges, interior and boundary bus counts.
PAPER_14 = {
    2: ([0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1], 3, [6, 3], [2, 3]),
    3: ([0, 0, 0, 0, 0, 1, 2, 2, 2, 1, 1, 1, 1, 1], 5, [3, 3, 1], [2, 3, 2]),
}


@pytest.fixture(scope="module")
def edges(splits):
    return splits["train"].edge_index_np, splits["train"].N


@pytest.mark.parametrize("K", [2, 3])
def test_spectral_partition_reproduces_the_paper(edges, K):
    pytest.importorskip("sklearn")
    ei, N = edges
    p = spectral_partition(ei, N, K)
    assignment, cut, interior, boundary = PAPER_14[K]
    assert p.assignment.tolist() == assignment and p.cut_edges == cut and p.K == K
    assert p.interior.sum(axis=1).tolist() == interior and p.boundary.sum(axis=1).tolist() == boundary
    assert (p.interior | p.boundary).sum(axis=0).tolist() == [1] * N  # every bus in exactly one client


def test_one_client_holds_every_bus_without_clustering(edges):
    ei, N = edges
    p = spectral_partition(ei, N, 1)
    assert p.K == 1 and not p.assignment.any() and p.cut_edges == 0 and p.interior.all()
    for bad in (0, N + 1):
        with pytest.raises(ValueError, match="K must be between"):
            spectral_partition(ei, N, bad)


def test_halo_adds_other_clients_rings_after_the_own_buses(edges):
    ei, N = edges
    p = partition_from_assignment(PAPER_14[2][0], ei)
    nodes, owned = compute_nodes(p, ei, 1, halo=0)
    assert owned == 6 and nodes.tolist() == p.owned(1).tolist()
    nodes, owned = compute_nodes(p, ei, 1, halo=1)
    ring = nodes[owned:]
    assert len(ring) and (p.assignment[ring] == 0).all()  # only the other client's buses
    A = np.zeros((N, N), bool)
    A[ei[0], ei[1]] = A[ei[1], ei[0]] = True
    assert all(A[b, p.owned(1)].any() for b in ring)  # each one hop from client 1
    assert nodes[:owned].tolist() == p.owned(1).tolist()


def test_halo_depth_two_reaches_further_and_stops_when_nothing_is_left():
    # a path 0-1-2-3-4 with client 0 = {0}: depth 2 adds 1 then 2; depth 9 stops at 4
    A = np.zeros((5, 5))
    for u in range(4):
        A[u, u + 1] = A[u + 1, u] = 1
    a = np.array([0, 1, 1, 1, 1])
    assert halo_nodes(a, A, 0, 2)[0].tolist() == [0, 1, 2]
    assert halo_nodes(a, A, 0, 9)[0].tolist() == [0, 1, 2, 3, 4]


def test_fedavg_is_exact_for_one_client_and_weights_by_count():
    w = np.random.default_rng(0).normal(size=(4, 3)).astype(np.float32)
    assert np.array_equal(fedavg([w], [7]), w) and fedavg([w], [7]).dtype == np.float32
    a, b = np.ones(3, np.float32), np.full(3, 4.0, np.float32)
    assert np.allclose(fedavg([a, b], [1, 3]), 3.25)
    with pytest.raises(ValueError, match="one weight per client"):
        fedavg([a], [1, 1])
    with pytest.raises(ValueError, match="one weight per client"):
        fedavg([], [])
    with pytest.raises(ValueError, match="positive"):
        fedavg([a, b], [1, 0])
    big = fedavg([a, b], [1e308, 1e308])  # finite weights near the float limit still average
    assert np.allclose(big, 2.5)
    for bad in (np.nan, np.inf):
        with pytest.raises(ValueError, match="finite and positive"):
            fedavg([a, b], [1, bad])
    with pytest.raises(ValueError, match="shape"):
        fedavg([np.ones((1, 3)), np.ones(3)], [1, 1])  # broadcastable is not the same parameter


def test_attackable_affinity_raises_edges_at_attackable_buses():
    A = np.ones((3, 3)) - np.eye(3)
    W = attackable_affinity(A, np.array([True, False, False]), heavy=4.0)
    assert W[0, 1] == pytest.approx(2.0) and W[1, 2] == pytest.approx(1.0) and W[0, 0] == 0


def test_halo_reaches_past_the_clients_own_buses():
    """Every own bus starts the search, so owned -> owned -> foreign is one hop from the client."""
    A = np.zeros((3, 3))
    A[0, 1] = A[1, 0] = A[1, 2] = A[2, 1] = 1
    assert halo_nodes(np.array([0, 0, 1]), A, 0, 1)[0].tolist() == [0, 1, 2]
    assert halo_nodes(np.array([0, 0, 1]), A, 0, 2)[0].tolist() == [0, 1, 2]


def test_partition_and_affinity_inputs_are_checked(edges):
    ei, N = edges
    with pytest.raises(ValueError, match="one integer client per bus"):
        partition_from_assignment(np.zeros(N - 1, int), ei)
    with pytest.raises(ValueError, match="one integer client per bus"):
        partition_from_assignment(np.zeros(0, int), np.zeros((2, 0), int))
    with pytest.raises(ValueError, match="none empty"):
        partition_from_assignment(np.r_[np.zeros(N - 1, int), 2], ei)  # client 1 has no bus
    with pytest.raises(ValueError, match="attackable mask"):
        attackable_affinity(np.eye(3), np.array([True, False]))
    for bad in (np.ones((N, 1), bool), np.ones(1, bool)):  # neither broadcasts silently
        with pytest.raises(ValueError, match="one flag per bus"):
            spectral_partition(ei, N, 1, attackable=bad)
    assert spectral_partition(ei, N, 1, attackable=np.ones(N, bool)).attackable_boundary == 0


def test_every_formula_refuses_malformed_input():
    from fdia_graph.formulas.federated import channel_moments, cut_edge_count, interior_boundary, pool_moments

    A = np.zeros((3, 3))
    A[0, 1] = A[1, 0] = 1
    a = np.array([0, 0, 1])
    for bad in (np.inf, np.nan, 0.0, 1e300):
        with pytest.raises(ValueError, match="heavy"):
            attackable_affinity(A, np.ones(3, bool), heavy=bad)
    with pytest.raises(ValueError, match="square"):
        attackable_affinity(np.ones(3), np.ones(3, bool))
    with pytest.raises(ValueError, match="one scalar weight"):
        fedavg([np.ones(2), np.ones(2)], [[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="owns no bus"):
        halo_nodes(a, A, 5, 1)  # a client id that is not in the assignment
    with pytest.raises(ValueError, match="depth"):
        halo_nodes(a, A, 0, -1)
    for f in (lambda: interior_boundary(a[:2], A, 2), lambda: cut_edge_count(a, A[:2])):
        with pytest.raises(ValueError, match="adjacency"):
            f()
    with pytest.raises(ValueError, match="exactly the clients"):
        interior_boundary(np.array([0, 1]), np.zeros((2, 2)), 1)  # K drops client 1
    with pytest.raises(ValueError, match="non-empty"):
        channel_moments(np.zeros((0, 3, 2)))
    with pytest.raises(ValueError, match="at least one part"):
        pool_moments([])


def test_topology_is_checked_and_one_bus_per_client_needs_no_clustering(edges):
    from fdia_graph.federated import bus_adjacency

    ei, N = edges
    for bad in (np.array([[0, -1]]).T, np.array([[0, N]]).T, np.zeros((3, 2), int), np.zeros((2, 2))):
        with pytest.raises(ValueError, match="edge_index"):
            bus_adjacency(bad, N)
    p = spectral_partition(ei, N, N)
    assert p.assignment.tolist() == list(range(N)) and p.interior.sum() + p.boundary.sum() == N


def test_hop_distance_and_the_halo_agree_with_a_hand_count():
    from fdia_graph.formulas.federated import hop_distance

    # 0-1-2-3 and 1-4, client 0 = {0, 1}: bus 2 and bus 4 are one hop away, bus 3 two
    A = np.zeros((5, 5))
    for u, v in ((0, 1), (1, 2), (2, 3), (1, 4)):
        A[u, v] = A[v, u] = 1
    a = np.array([0, 0, 1, 1, 2])
    assert hop_distance(A, np.array([0, 1])).tolist() == [0, 0, 1, 2, 1]
    assert halo_nodes(a, A, 0, 1)[0].tolist() == [0, 1, 2, 4]
    assert halo_nodes(a, A, 0, 2)[0].tolist() == [0, 1, 2, 4, 3]
    B = np.zeros((3, 3))
    assert hop_distance(B, np.array([0])).tolist() == [0, -1, -1]  # no path


# ---- the federated localizers ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def zs(timeline):
    import fdia_graph as fg

    z = dict(families=[0, 1, 2])
    return (
        fg.load(timeline, split="train", **z),
        fg.load(timeline, split="val", **z),
        fg.load(timeline, split="test", families=[0, 1, 2, 3, 4]),
    )


@pytest.mark.parametrize("name", ["mlp", "cnn"])
def test_one_client_is_the_centralized_fit_weight_for_weight(zs, name):
    torch = pytest.importorskip("torch")
    from fdia_graph.federated.localizer import FedBusCNN, FedBusMLP
    from fdia_graph.localization import BusCNN, BusMLP

    was = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        tr, va, te = zs
        central, fed = (BusMLP, FedBusMLP) if name == "mlp" else (BusCNN, FedBusCNN)
        c = central(epochs=4, device="cpu").fit(tr, val=va)
        f = fed(K=1, rounds=2, local_epochs=2, grad_clip=None, device="cpu").fit(tr, val=va)
        cs, fs = c.net.state_dict(), f.net.state_dict()
        assert cs.keys() == fs.keys() and all(torch.equal(cs[k], fs[k]) for k in cs)
        assert np.array_equal(c.scores(te), f.scores(te)) and c.tau == f.tau
    finally:
        torch.use_deterministic_algorithms(was)


def test_two_clients_train_score_and_log_their_rounds(zs):
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    from fdia_graph.federated.aggregate import state_bytes
    from fdia_graph.federated.localizer import FedBusCNN

    tr, va, te = zs
    loc = FedBusCNN(K=2, rounds=3, local_epochs=1, device="cpu").fit(tr, val=va)
    assert len(loc.history) == 3 and len(loc.history[0].loss_per_client) == 2
    assert loc.history[0].bytes_up == 2 * state_bytes(loc.net.state_dict())
    import torch

    final = loc.net.state_dict()  # every client holds the broadcast average after the last round
    assert all(all(torch.equal(v, c.net.state_dict()[k]) for k, v in final.items()) for c in loc._clients)
    s = loc.scores(te)
    assert s.shape == (len(te), te.N) and np.all((s >= 0) & (s <= 1))
    rep = loc.score(te)
    assert "As" in rep and "Ar" in rep and loc.tau is not None


def test_local_power_balance_changes_only_buses_on_foreign_metered_branches(zs):
    pytest.importorskip("torch")
    from fdia_graph.federated import partition_from_assignment
    from fdia_graph.federated.localizer import FedBusMLP

    tr, _, _ = zs
    ei = tr.edge_index_np
    part = partition_from_assignment(PAPER_14[2][0], ei)
    loc = FedBusMLP(K=2, partition=part)
    loc._part = part
    d = tr.export(loc._fields())
    glob, local = loc._features(d), loc._client_features(d, 0)
    changed = np.flatnonzero(np.abs(glob - local).max(axis=(0, 2)) > 0)
    foreign = part.assignment[ei[0]] != 0  # branches metered at the other client's end
    touched = set(ei[0][foreign].tolist()) | set(ei[1][foreign].tolist())
    assert len(changed) and set(changed.tolist()) <= touched
    assert np.array_equal(
        np.delete(glob, list(touched), axis=1)[..., :8], np.delete(local, list(touched), axis=1)[..., :8]
    )


def test_federated_constructor_checks():
    pytest.importorskip("torch")
    from fdia_graph.federated.localizer import FedBusMLP

    with pytest.raises(ValueError, match="rounds x local_epochs"):
        FedBusMLP(epochs=5)
    with pytest.raises(ValueError, match="kcl"):
        FedBusMLP(kcl="bogus")
    for bad in (dict(K=2.5), dict(rounds=True), dict(halo=1.0)):
        with pytest.raises(ValueError, match="FederatedSettings.(K|rounds|halo) must be an integer"):
            FedBusMLP(**bad)
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="grad_clip"):
            FedBusMLP(grad_clip=bad)
    three = partition_from_assignment(np.array([0, 1, 2]), np.array([[0, 1], [1, 2]]))
    with pytest.raises(ValueError, match="3 clients but K=2"):
        FedBusMLP(K=2, partition=three)


def test_a_partition_of_another_grid_is_refused(zs):
    pytest.importorskip("torch")
    from fdia_graph.federated.localizer import FedBusMLP

    tr, va, _ = zs
    small = partition_from_assignment(np.array([0, 0, 1]), np.array([[0, 1], [1, 2]]))
    from fdia_graph.models.federated import Partition

    N = tr.N
    odd = Partition(2, np.r_[np.zeros(N - 1, int), 5], np.zeros((2, N), bool), np.zeros((2, N), bool), 0)
    with pytest.raises(ValueError, match="number its clients 0..1"):
        FedBusMLP(K=2, partition=odd, rounds=1, local_epochs=1, device="cpu").fit(tr, val=va)
    with pytest.raises(ValueError, match="covers 3 buses"):
        FedBusMLP(K=2, partition=small, rounds=1, local_epochs=1, device="cpu").fit(tr, val=va)


def test_a_two_client_fit_is_reproducible_and_leaves_no_client_stream_shared(zs):
    """Each client trains on its own dropout stream, so the same seed gives the same weights, and
    clients do not draw from one shared stream (a dropout-free model gives the same answer)."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    from fdia_graph.federated.localizer import FedBusMLP

    was = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        tr, va, _ = zs
        kw = dict(K=2, rounds=2, local_epochs=1, device="cpu")
        a, b = FedBusMLP(**kw).fit(tr, val=va), FedBusMLP(**kw).fit(tr, val=va)
        assert all(
            torch.equal(x, y) for x, y in zip(a.net.state_dict().values(), b.net.state_dict().values())
        )
        c0, c1 = a._clients
        assert not torch.equal(c0.rng[0], c1.rng[0])  # two streams, not one
    finally:
        torch.use_deterministic_algorithms(was)


def test_a_halo_client_gets_context_features_but_only_its_own_labels(zs):
    pytest.importorskip("torch")
    from fdia_graph.federated import partition_from_assignment
    from fdia_graph.federated.localizer import FedBusMLP

    tr, va, _ = zs
    part = partition_from_assignment(PAPER_14[2][0], tr.edge_index_np)
    loc = FedBusMLP(K=2, partition=part, halo=1, rounds=1, local_epochs=1, device="cpu").fit(tr, val=va)
    for c in loc._clients:
        assert len(c.nodes) > c.owned  # the halo is there as features
        assert tuple(c.Xt.shape[1:2]) == (len(c.nodes),) and tuple(c.Yt.shape[1:]) == (c.owned,)


# ---- federated state estimation -----------------------------------------------------------------------


def test_one_region_is_the_subspace_prior_and_two_are_block_diagonal(splits, edges):
    pytest.importorskip("pandapower")
    from fdia_graph.federated import RegionalPrior
    from fdia_graph.se import SubspacePrior

    ei, N = edges
    tr, te = splits["train"], splits["test"]
    one = partition_from_assignment(np.zeros(N, int), ei)
    base = SubspacePrior(rank_frac=0.5).fit(tr)
    reg = RegionalPrior(one, rank_frac=0.5).fit(tr)
    assert np.array_equal(base.VK, reg.VK) and np.array_equal(base.estimate(te), reg.estimate(te))

    two = RegionalPrior(partition_from_assignment(PAPER_14[2][0], ei), rank_frac=0.5).fit(tr)
    V = two.VK
    assert np.allclose(V.T @ V, np.eye(V.shape[1]), atol=1e-10)
    c0, c1 = two.state_columns(0), two.state_columns(1)
    assert len(np.intersect1d(c0, c1)) == 0 and len(c0) + len(c1) == V.shape[0]
    k0 = int(np.linalg.matrix_rank(V[c0]))
    assert not V[np.ix_(c0, np.arange(k0, V.shape[1]))].any() and not V[np.ix_(c1, np.arange(k0))].any()
    assert np.isfinite(two.score(te)["geo"]["angle_mae_deg"])


def test_a_federated_localizer_gates_the_estimator(zs, splits):
    pytest.importorskip("pandapower")
    pytest.importorskip("torch")
    from fdia_graph.federated.localizer import FedBusMLP
    from fdia_graph.se import GatedPrior

    tr, va, _ = zs
    gate = FedBusMLP(K=2, rounds=2, local_epochs=1, device="cpu").fit(tr, val=va)
    est = GatedPrior(gate=gate, rank_frac=0.5, reweight="huber").fit(splits["train"])
    assert np.isfinite(est.score(splits["test"])["geo"]["angle_mae_deg"])


def test_block_diagonal_basis_checks_its_blocks():
    from fdia_graph.formulas.federated import block_diagonal_basis

    V = block_diagonal_basis([(np.array([0, 2]), np.eye(2)), (np.array([1]), np.ones((1, 1)))], 3)
    assert V.shape == (3, 3) and V[1, 2] == 1 and V[0, 0] == 1 and V[2, 1] == 1
    with pytest.raises(ValueError, match="disjoint"):
        block_diagonal_basis([(np.array([0, 1]), np.eye(2)), (np.array([1]), np.ones((1, 1)))], 3)
    for cols in (np.array([0, 5]), np.array([0.0, 1.0]), np.array([[0, 1]])):
        with pytest.raises(ValueError, match="integer state columns"):
            block_diagonal_basis([(cols, np.eye(2))], 3)
    with pytest.raises(ValueError, match="one row per state column"):
        block_diagonal_basis([(np.array([0, 1]), np.eye(3))], 3)
    for flat in (np.ones(2), np.float64(1.0)):  # not a [rows, K] basis
        with pytest.raises(ValueError, match="one row per state column"):
            block_diagonal_basis([(np.array([0, 1]), flat)], 3)


def test_a_regional_prior_refuses_a_partition_with_gaps(splits):
    pytest.importorskip("pandapower")
    from fdia_graph.federated import RegionalPrior
    from fdia_graph.models.federated import Partition

    N = splits["train"].N
    gap = Partition(2, np.r_[0, np.full(N - 1, 2)], np.zeros((2, N), bool), np.zeros((2, N), bool), 0)
    with pytest.raises(ValueError, match="number its clients 0..1"):
        RegionalPrior(gap).fit(splits["train"])


def test_check_partition_wants_integer_labels(splits):
    from fdia_graph.federated import check_partition
    from fdia_graph.models.federated import Partition

    N = splits["train"].N
    z = np.zeros((2, N), bool)
    for a in (np.r_[np.zeros(N - 1), 1.0], np.zeros((1, N), int)):
        with pytest.raises(ValueError, match="1-D integer array"):
            check_partition(Partition(2, a, z, z, 0), N)


def test_the_jacobian_feature_set_federates_with_one_central_block(zs):
    """full14+jac: the 14 channels stay per client (local KCL), the Jacobian block is the central
    estimator's, identical for every client; one client equals the centralized CNN."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("pandapower")
    from fdia_graph.federated.localizer import FedBusCNN
    from fdia_graph.localization import BusCNN

    was = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        tr, va, te = zs
        c = BusCNN(features="full14+jac", epochs=2, device="cpu").fit(tr, val=va)
        f = FedBusCNN(K=1, rounds=2, local_epochs=1, grad_clip=None, features="full14+jac", device="cpu").fit(
            tr, val=va
        )
        assert np.array_equal(c.scores(te), f.scores(te))
        two = FedBusCNN(K=2, rounds=1, local_epochs=1, features="full14+jac", device="cpu").fit(tr, val=va)
        d = te.export(two._fields())
        a, b = two._client_features(d, 0), two._client_features(d, 1)
        assert a.shape[-1] == 22 and np.array_equal(a[..., 14:], b[..., 14:])  # one Jacobian block
        assert "As" in two.score(te)
    finally:
        torch.use_deterministic_algorithms(was)


def test_the_jacobian_only_set_federates_and_refuses_per_unit(zs, timeline):
    """features="jac": every client sees only the central 8-channel block; a per-unit view is refused."""
    pytest.importorskip("torch")
    pytest.importorskip("pandapower")
    import fdia_graph as fg
    from fdia_graph.federated.localizer import FedBusMLP

    tr, va, te = zs
    m = FedBusMLP(K=2, rounds=1, local_epochs=1, features="jac", device="cpu").fit(tr, val=va)
    d = te.export(m._fields())
    a, b = m._client_features(d, 0), m._client_features(d, 1)
    assert a.shape[-1] == 8 and np.array_equal(a, b)
    assert "As" in m.score(te)
    assert np.array_equal(m._client_features(d, 0, m._central(d)), a)  # handed in = computed in place
    with pytest.raises(ValueError, match="physical"):
        m.score(fg.load(timeline, split="test", units="pu"))


def test_the_previous_swing_set_federates_client_locally(zs):
    """full14+prev: the previous frame's swing is each bus's own reading, so every client builds it;
    one client is the centralized fit, two clients fit and score the unseen families."""
    torch = pytest.importorskip("torch")
    from fdia_graph.federated.localizer import FedBusCNN
    from fdia_graph.localization import BusCNN

    was = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        tr, va, te = zs
        c = BusCNN(features="full14+prev", epochs=2, device="cpu").fit(tr, val=va)
        f = FedBusCNN(
            K=1, rounds=2, local_epochs=1, grad_clip=None, features="full14+prev", device="cpu"
        ).fit(tr, val=va)
        assert np.array_equal(c.scores(te), f.scores(te))
        two = FedBusCNN(K=2, rounds=1, local_epochs=1, features="full14+prev", device="cpu").fit(tr, val=va)
        assert two._client_features(two._pull(te), 0).shape[-1] == 16 and "As" in two.score(te)
    finally:
        torch.use_deterministic_algorithms(was)
