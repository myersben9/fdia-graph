"""Generate Wu-style load multipliers for any bus-system size."""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd


UTILITY_NAMES = [
    "WEST",
    "GENESE",
    "CENTRL",
    "NORTH",
    "MHK_VL",
    "CAPITL",
    "HUD_VL",
    "MILLWD",
    "DUNWOD",
    "NYC",
    "LONGIL",
]


def assign_utility(bus_id: int, number_of_buses: int) -> tuple[int, str]:
    """
    Assign each bus to a synthetic federated-learning client.

    This is a temporary test-system assignment. It can later be
    replaced by a real geographic bus-to-utility mapping.
    """
    number_of_utilities = min(number_of_buses, len(UTILITY_NAMES))
    utility_id = (bus_id - 1) % number_of_utilities

    return utility_id, UTILITY_NAMES[utility_id]


def generate_load_series(
    number_of_buses: int,
    number_of_snapshots: int,
    interval_minutes: int,
    seed: int,
) -> pd.DataFrame:
    """
    Generate system and individual bus load multipliers.

    Wu method:
        system_factor ~ Uniform(0.90, 1.00)
        bus_fluctuation ~ Normal(0.00, 0.03)

        load_multiplier = system_factor + bus_fluctuation
    """
    rng = np.random.default_rng(seed)

    timestamps = pd.date_range(
        start="2026-01-01 00:00:00",
        periods=number_of_snapshots,
        freq=f"{interval_minutes}min",
    )

    # One system-wide operating level per snapshot.
    system_factors = rng.uniform(
        low=0.90,
        high=1.00,
        size=number_of_snapshots,
    )

    records = []

    for snapshot, timestamp in enumerate(timestamps):
        system_factor = system_factors[snapshot]

        # Every bus receives its own individual fluctuation.
        bus_fluctuations = rng.normal(
            loc=0.0,
            scale=0.03,
            size=number_of_buses,
        )

        for bus_index in range(number_of_buses):
            bus_id = bus_index + 1

            utility_id, utility_name = assign_utility(
                bus_id=bus_id,
                number_of_buses=number_of_buses,
            )

            bus_fluctuation = bus_fluctuations[bus_index]

            # This multiplier will be applied to base P and Q in Task 2.
            load_multiplier = system_factor + bus_fluctuation

            records.append(
                {
                    "snapshot": snapshot,
                    "timestamp": timestamp,
                    "bus_id": bus_id,
                    "utility_id": utility_id,
                    "utility_name": utility_name,
                    "system_factor": system_factor,
                    "bus_fluctuation": bus_fluctuation,
                    "load_multiplier": load_multiplier,
                }
            )

    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Wu-style bus load multipliers."
    )

    parser.add_argument(
        "--buses",
        type=int,
        default=140,
        help="Number of buses",
    )
    parser.add_argument(
        "--snapshots",
        type=int,
        default=288,
        help="Number of load snapshots",
    )
    parser.add_argument(
        "--interval-minutes",
        type=int,
        default=5,
        help="Minutes between snapshots",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed",
    )

    args = parser.parse_args()

    if args.buses <= 0:
        raise ValueError("--buses must be greater than zero")

    if args.snapshots <= 0:
        raise ValueError("--snapshots must be greater than zero")

    load_series = generate_load_series(
        number_of_buses=args.buses,
        number_of_snapshots=args.snapshots,
        interval_minutes=args.interval_minutes,
        seed=args.seed,
    )

    output_directory = Path("generated_data")
    output_directory.mkdir(exist_ok=True)

    output_path = (
        output_directory
        / f"wu_load_series_{args.buses}_bus.csv"
    )

    load_series.to_csv(output_path, index=False)

    print("Load-series generation complete")
    print(f"Bus count: {args.buses}")
    print(f"Snapshots: {args.snapshots}")
    print(f"Rows: {len(load_series):,}")
    print(f"Saved to: {output_path}")
    print()
    print(load_series.head(10))
    print()
    print("Multiplier summary:")
    print(load_series["load_multiplier"].describe())


if __name__ == "__main__":
    main()