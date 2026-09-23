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
    with pytest.raises(ValueError, match="K must be"):
        spectral_partition(ei, N, 0)


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


def test_attackable_affinity_raises_edges_at_attackable_buses():
    A = np.ones((3, 3)) - np.eye(3)
    W = attackable_affinity(A, np.array([True, False, False]), heavy=4.0)
    assert W[0, 1] == pytest.approx(2.0) and W[1, 2] == pytest.approx(1.0) and W[0, 0] == 0
