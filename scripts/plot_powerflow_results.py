"""Create plots from the timeline AC power-flow results."""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_results(
    results_directory: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and verify Task 3 result files."""
    bus_path = results_directory / "bus_results.csv"
    branch_path = results_directory / "branch_results.csv"
    convergence_path = (
        results_directory / "convergence.csv"
    )

    for path in [
        bus_path,
        branch_path,
        convergence_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required result file was not found: {path}"
            )

    bus_results = pd.read_csv(
        bus_path,
        parse_dates=["timestamp"],
    )

    branch_results = pd.read_csv(
        branch_path,
        parse_dates=["timestamp"],
    )

    convergence = pd.read_csv(
        convergence_path,
        parse_dates=["timestamp"],
    )

    return bus_results, branch_results, convergence


def save_figure(
    output_path: Path,
) -> None:
    """Apply final formatting and save a figure."""
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Created: {output_path}")


def plot_total_system_load(
    bus_results: pd.DataFrame,
    output_directory: Path,
) -> None:
    """Plot total real and reactive load over time."""
    system_load = (
        bus_results.groupby(
            ["snapshot", "timestamp"],
            as_index=False,
        )[["p_mw", "q_mvar"]]
        .sum()
        .sort_values("snapshot")
    )

    plt.figure(figsize=(12, 6))

    plt.plot(
        system_load["timestamp"],
        system_load["p_mw"],
        label="Total real power P",
        linewidth=2,
        color="tab:blue",
    )

    plt.plot(
        system_load["timestamp"],
        system_load["q_mvar"],
        label="Total reactive power Q",
        linewidth=2,
        color="tab:orange",
    )

    plt.title("Total System Load Over Time")
    plt.xlabel("Time")
    plt.ylabel("Power")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(rotation=30)

    save_figure(
        output_directory / "total_system_load.png"
    )


def plot_bus_voltage_timelines(
    bus_results: pd.DataFrame,
    output_directory: Path,
) -> None:
    """Plot voltage magnitude over time for every bus."""
    plt.figure(figsize=(13, 7))

    for bus_id, bus_data in bus_results.groupby(
        "bus_id"
    ):
        bus_data = bus_data.sort_values("snapshot")

        plt.plot(
            bus_data["timestamp"],
            bus_data["voltage_pu"],
            label=f"Bus {int(bus_id)}",
            linewidth=1.3,
        )

    plt.axhline(
        1.05,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Upper limit: 1.05 pu",
    )

    plt.axhline(
        0.95,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Lower limit: 0.95 pu",
    )

    plt.title("Bus Voltage Magnitudes Over Time")
    plt.xlabel("Time")
    plt.ylabel("Voltage magnitude (pu)")
    plt.grid(True, alpha=0.3)
    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=8,
    )
    plt.xticks(rotation=30)

    save_figure(
        output_directory / "bus_voltage_timelines.png"
    )


def plot_voltage_heatmap(
    bus_results: pd.DataFrame,
    output_directory: Path,
) -> None:
    """Create a heatmap of voltage across buses and time."""
    voltage_matrix = bus_results.pivot(
        index="bus_id",
        columns="snapshot",
        values="voltage_pu",
    )

    voltage_matrix = voltage_matrix.sort_index()

    plt.figure(figsize=(14, 7))

    image = plt.imshow(
        voltage_matrix.to_numpy(),
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="viridis",
    )

    colorbar = plt.colorbar(image)
    colorbar.set_label("Voltage magnitude (pu)")

    plt.title("Bus Voltage Heatmap")
    plt.xlabel("Snapshot")
    plt.ylabel("Bus")

    bus_count = len(voltage_matrix.index)

    plt.yticks(
        ticks=np.arange(bus_count),
        labels=voltage_matrix.index.astype(int),
    )

    save_figure(
        output_directory / "voltage_heatmap.png"
    )


def plot_selected_snapshot(
    bus_results: pd.DataFrame,
    output_directory: Path,
    selected_snapshot: int,
) -> None:
    """Plot voltage magnitude across buses at one snapshot."""
    snapshot_data = bus_results[
        bus_results["snapshot"] == selected_snapshot
    ].copy()

    if snapshot_data.empty:
        raise ValueError(
            f"Snapshot {selected_snapshot} does not exist."
        )

    snapshot_data = snapshot_data.sort_values(
        "bus_id"
    )

    timestamp = snapshot_data["timestamp"].iloc[0]

    plt.figure(figsize=(11, 6))

    plt.plot(
        snapshot_data["bus_id"],
        snapshot_data["voltage_pu"],
        marker="o",
        linewidth=2,
        color="tab:blue",
    )

    plt.axhline(
        1.05,
        color="red",
        linestyle="--",
        label="Upper limit: 1.05 pu",
    )

    plt.axhline(
        0.95,
        color="red",
        linestyle="--",
        label="Lower limit: 0.95 pu",
    )

    plt.title(
        f"Bus Voltage Profile - Snapshot "
        f"{selected_snapshot}\n{timestamp}"
    )
    plt.xlabel("Bus number")
    plt.ylabel("Voltage magnitude (pu)")
    plt.xticks(snapshot_data["bus_id"])
    plt.grid(True, alpha=0.3)
    plt.legend()

    save_figure(
        output_directory
        / f"voltage_snapshot_{selected_snapshot}.png"
    )


def plot_maximum_branch_loading(
    branch_results: pd.DataFrame,
    output_directory: Path,
) -> None:
    """Plot the maximum line loading at each snapshot."""
    valid_loading = branch_results.dropna(
        subset=["loading_percent"]
    )

    if valid_loading.empty:
        print(
            "Skipped branch-loading plot because the case "
            "does not contain usable branch ratings."
        )
        return

    maximum_loading = (
        valid_loading.groupby(
            ["snapshot", "timestamp"],
            as_index=False,
        )["loading_percent"]
        .max()
        .sort_values("snapshot")
    )

    plt.figure(figsize=(12, 6))

    plt.plot(
        maximum_loading["timestamp"],
        maximum_loading["loading_percent"],
        linewidth=2,
        color="tab:red",
        label="Most heavily loaded branch",
    )

    plt.axhline(
        100.0,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="100% branch rating",
    )

    plt.title("Maximum Branch Loading Over Time")
    plt.xlabel("Time")
    plt.ylabel("Loading (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(rotation=30)

    save_figure(
        output_directory
        / "maximum_branch_loading.png"
    )


def plot_voltage_extremes(
    bus_results: pd.DataFrame,
    output_directory: Path,
) -> None:
    """Plot minimum and maximum bus voltage per snapshot."""
    voltage_extremes = (
        bus_results.groupby(
            ["snapshot", "timestamp"],
            as_index=False,
        )["voltage_pu"]
        .agg(["min", "max"])
        .reset_index()
        .sort_values("snapshot")
    )

    plt.figure(figsize=(12, 6))

    plt.plot(
        voltage_extremes["timestamp"],
        voltage_extremes["max"],
        label="Maximum bus voltage",
        linewidth=2,
        color="tab:green",
    )

    plt.plot(
        voltage_extremes["timestamp"],
        voltage_extremes["min"],
        label="Minimum bus voltage",
        linewidth=2,
        color="tab:purple",
    )

    plt.axhline(
        1.05,
        color="red",
        linestyle="--",
        linewidth=1.3,
        label="Upper limit: 1.05 pu",
    )

    plt.axhline(
        0.95,
        color="red",
        linestyle="--",
        linewidth=1.3,
        label="Lower limit: 0.95 pu",
    )

    plt.title("Minimum and Maximum Bus Voltage")
    plt.xlabel("Time")
    plt.ylabel("Voltage magnitude (pu)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(rotation=30)

    save_figure(
        output_directory / "voltage_extremes.png"
    )


def print_convergence_summary(
    convergence: pd.DataFrame,
) -> None:
    """Print the number of converged snapshots."""
    values = (
        convergence["converged"]
        .astype(str)
        .str.lower()
    )

    converged_count = values.isin(
        ["true", "1"]
    ).sum()

    total_count = len(convergence)

    print()
    print(
        f"Converged snapshots: "
        f"{converged_count}/{total_count}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create plots from timeline power-flow results."
        )
    )

    parser.add_argument(
        "--results",
        default="generated_data/powerflow_14_bus",
        help="Task 3 results directory",
    )

    parser.add_argument(
        "--output",
        default="generated_data/powerflow_14_bus/plots",
        help="Directory where plot images will be saved",
    )

    parser.add_argument(
        "--snapshot",
        type=int,
        default=144,
        help="Snapshot used for the voltage-profile plot",
    )

    args = parser.parse_args()

    results_directory = Path(args.results)
    output_directory = Path(args.output)

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        bus_results,
        branch_results,
        convergence,
    ) = load_results(results_directory)

    print_convergence_summary(convergence)
    print()

    plot_total_system_load(
        bus_results,
        output_directory,
    )

    plot_bus_voltage_timelines(
        bus_results,
        output_directory,
    )

    plot_voltage_heatmap(
        bus_results,
        output_directory,
    )

    plot_selected_snapshot(
        bus_results,
        output_directory,
        args.snapshot,
    )

    plot_maximum_branch_loading(
        branch_results,
        output_directory,
    )

    plot_voltage_extremes(
        bus_results,
        output_directory,
    )

    print()
    print("All plots finished.")
    print(f"Plots saved in: {output_directory}")


if __name__ == "__main__":
    main()