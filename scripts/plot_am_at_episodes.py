"""Compare complete Am and At attack episodes over time."""

from os import system
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import h5py

import fdia_graph as fg


SYSTEMS = (
    "ieee14",
    "ieee30",
    "ieee57",
    "ieee89",
    "ieee118",
    "ieee145",
    "ieee200",
    "ieee300",
)
CONTEXT_FRAMES = 10

# FDIA family numbers
AT_FAMILY = 5
AM_FAMILY = 7


def find_episode(episodes, family_code):
    """Find the longest episode belonging to the requested family."""
    matching = np.flatnonzero(episodes.family == family_code)

    if matching.size == 0:
        raise RuntimeError(f"No episodes found for family {family_code}")

    # Select the longest episode instead of a one-frame record.
    lengths = episodes.length[matching]
    return int(matching[np.argmax(lengths)])


def episode_data(dataset, episode_index):
    """Extract one complete episode with context before and after it."""
    episodes = dataset.episodes

    onset = int(episodes.onset[episode_index])
    length = int(episodes.length[episode_index])
    buses = np.asarray(episodes.buses[episode_index], dtype=int)

    start = max(0, onset - CONTEXT_FRAMES)

    with h5py.File(dataset.path, "r") as file:
        total_frames = file["data/node_x"].shape[0]
        stop = min(total_frames, onset + length + CONTEXT_FRAMES)

        observed = file["data/node_x"][start:stop]
        clean = file["clean/node_clean"][start:stop]

    # Column 1 is active-power injection.
    attack_delta = observed[:, :, 1] - clean[:, :, 1]

    # Choose the attacked bus showing the largest change during the episode.
    local_start = onset - start
    local_stop = local_start + length

    bus_strength = np.max(
        np.abs(attack_delta[local_start:local_stop, buses]),
        axis=0,
    )
    bus = int(buses[np.argmax(bus_strength)])

    # Make attack onset equal to time zero.
    time = np.arange(start, stop) - onset

    return {
        "time": time,
        "observed": observed[:, bus, 1],
        "clean": clean[:, bus, 1],
        "delta": attack_delta[:, bus],
        "bus": bus,
        "length": length,
        "onset": onset,
    }


def plot_system(system):
    print(f"Loading {system} in full time order...")
    dataset = fg.load(system, order="time")

    at_index = find_episode(dataset.episodes, AT_FAMILY)
    am_index = find_episode(dataset.episodes, AM_FAMILY)

    at = episode_data(dataset, at_index)
    am = episode_data(dataset, am_index)

    print(
        f"At episode: onset={at['onset']}, "
        f"length={at['length']}, bus={at['bus'] + 1}"
    )
    print(
        f"Am episode: onset={am['onset']}, "
        f"length={am['length']}, bus={am['bus'] + 1}"
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11, 7),
        sharex="col",
        constrained_layout=True,
    )

    episodes = [("At", at), ("Am", am)]

    for row, (family_name, data) in enumerate(episodes):
        time = data["time"]
        end = data["length"] - 1

        # Left: measured signal compared with the clean signal.
        axes[row, 0].plot(
            time,
            data["clean"],
            color="black",
            linewidth=1.8,
            label="Clean",
        )
        axes[row, 0].plot(
            time,
            data["observed"],
            color="tab:red",
            linewidth=1.5,
            label="Observed",
        )
        axes[row, 0].axvspan(
            0,
            end,
            color="tab:red",
            alpha=0.12,
            label="Attack interval",
        )
        axes[row, 0].set_ylabel("Active-power injection")
        axes[row, 0].set_title(
            f"{family_name}: observed vs. clean, bus {data['bus'] + 1}"
        )
        axes[row, 0].grid(alpha=0.25)
        axes[row, 0].legend(loc="best")

        # Right: exact change introduced by the attack.
        axes[row, 1].plot(
            time,
            data["delta"],
            color="tab:blue",
            linewidth=1.8,
        )
        axes[row, 1].axhline(0, color="black", linewidth=0.8)
        axes[row, 1].axvspan(
            0,
            end,
            color="tab:red",
            alpha=0.12,
        )
        axes[row, 1].set_ylabel("Attack change")
        axes[row, 1].set_title(f"{family_name}: attack contribution")
        axes[row, 1].grid(alpha=0.25)

    axes[1, 0].set_xlabel("Time relative to attack onset")
    axes[1, 1].set_xlabel("Time relative to attack onset")

    fig.suptitle(
        f"Am and At attack episodes — {system}",
        fontsize=14,
    )

    output_directory = Path("docs") / "figures"
    output_directory.mkdir(parents=True, exist_ok=True)

    png_path = output_directory / f"{system}_am_at_time_comparison.png"
    pdf_path = output_directory / f"{system}_am_at_time_comparison.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")

    plt.close(fig)

def main():
    for system in SYSTEMS:
        try:
            plot_system(system)
        except Exception as error:
            print(f"FAILED {system}: {error}")

    print("Finished processing all IEEE systems.")
if __name__ == "__main__":
    main()