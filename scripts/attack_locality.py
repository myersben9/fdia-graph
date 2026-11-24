"""Measure how local each attack family remains in the power-grid graph."""

from collections import deque

import numpy as np
from torch_geometric import data
import fdia_graph as fg
import h5py

SYSTEMS = [
    "ieee14",
    "ieee30",
    "ieee57",
    "ieee89",
    "ieee118",
    "ieee145",
    "ieee200",
    "ieee300",
]

TOLERANCE = 1e-10

FAMILY_NAMES = {
    1: "Aq",
    2: "Ad",
    3: "As",
    4: "Ar",
    5: "At",
    6: "Al",
    7: "Am",
}


def shortest_hops(edge_index, number_of_buses):
    """Return the shortest number of graph hops between every pair of buses."""
    neighbors = [[] for _ in range(number_of_buses)]

    for source, destination in edge_index.T:
        source = int(source)
        destination = int(destination)

        neighbors[source].append(destination)
        neighbors[destination].append(source)

    distances = np.full(
        (number_of_buses, number_of_buses),
        np.inf,
        dtype=float,
    )

    for start in range(number_of_buses):
        distances[start, start] = 0
        queue = deque([start])

        while queue:
            current = queue.popleft()

            for neighbor in neighbors[current]:
                if np.isinf(distances[start, neighbor]):
                    distances[start, neighbor] = (
                        distances[start, current] + 1
                    )
                    queue.append(neighbor)

    return distances


def main():
    ds = fg.load(SYSTEM, order="time")
    data = ds.export(["y", "family"])

    attacked = np.asarray(data["y"], dtype=bool)
    family = np.asarray(data["family"], dtype=int)
    edge_index = np.asarray(ds.edge_index, dtype=int)

    h5 = h5py.File(ds.path, "r")

    node_tamper = h5["attack/node_tamper"][:].astype(bool)
    edge_tamper = h5["attack/edge_tamper"][:].astype(bool)

    h5.close()
    
    
    number_of_buses = attacked.shape[1]
    distances = shortest_hops(edge_index, number_of_buses)

    print(f"\nAttack-locality table for {SYSTEM}")
    print(
        f"{'Family':<8}"
        f"{'Frames':>10}"
        f"{'Mean node meters':>20}"
        f"{'Mean edge meters':>20}"
        f"{'Max hop':>12}"
    )

    for family_code, family_name in FAMILY_NAMES.items():
        frame_indices = np.flatnonzero(family == family_code)

        node_counts = []
        edge_counts = []
        family_max_hop = 0

        for frame in frame_indices:
            changed_node_meters = node_tamper[frame]
            changed_edge_meters = edge_tamper[frame]

            node_counts.append(np.count_nonzero(changed_node_meters))
            edge_counts.append(np.count_nonzero(changed_edge_meters))

            target_buses = np.flatnonzero(attacked[frame])

            tampered_buses = set(
                np.flatnonzero(changed_node_meters.any(axis=1))
            )

            changed_edges = np.flatnonzero(
                changed_edge_meters.any(axis=1)
            )

            for edge in changed_edges:
                tampered_buses.add(int(edge_index[0, edge]))
                tampered_buses.add(int(edge_index[1, edge]))

            for bus in tampered_buses:
                nearest_target = np.min(distances[target_buses, bus])
                family_max_hop = max(
                    family_max_hop,
                    int(nearest_target),
                )

        print(
            f"{family_name:<8}"
            f"{len(frame_indices):>10}"
            f"{np.mean(node_counts):>20.2f}"
            f"{np.mean(edge_counts):>20.2f}"
            f"{family_max_hop:>12}"
        )


if __name__ == "__main__":
    for SYSTEM in SYSTEMS:
        main()