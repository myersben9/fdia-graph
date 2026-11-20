"""Generate synthetic utility-level load timelines for federated learning."""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd


NYISO_CLIENTS = [
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


def assign_buses_to_utilities(number_of_buses: int) -> np.ndarray:
    """
    Give every bus to one synthetic utility/client.

    This is a temporary test-system mapping. It can later be replaced
    with geographic utility-boundary data.
    """
    number_of_clients = min(number_of_buses, len(NYISO_CLIENTS))

    return np.array(
        [bus_index % number_of_clients for bus_index in range(number_of_buses)]
    )


def generate_utility_profiles(
    number_of_steps: int,
    number_of_clients: int,
    seed: int,
) -> np.ndarray:
    """
    Generate a different full load timeline for every utility.

    Returns an array with shape:
        (number_of_steps, number_of_clients)
    """
    rng = np.random.default_rng(seed)
    time_index = np.arange(number_of_steps)
    minutes_per_day = 24 * 60

    profiles = np.zeros((number_of_steps, number_of_clients))

    for client in range(number_of_clients):
        # Give each utility a different daily pattern.
        phase_shift = rng.integers(-180, 181)
        amplitude = rng.uniform(0.10, 0.24)
        base_level = rng.uniform(0.88, 1.05)

        daily_cycle = amplitude * np.sin(
            2 * np.pi * (time_index + phase_shift) / minutes_per_day
            - np.pi / 2
        )

        # Slow regional variation.
        slow_cycle = 0.04 * np.sin(
            2 * np.pi * time_index / (minutes_per_day * 4)
            + rng.uniform(0, 2 * np.pi)
        )

        # Temporally correlated local variation.
        jitter = np.zeros(number_of_steps)

        for step in range(1, number_of_steps):
            jitter[step] = (
                0.94 * jitter[step - 1]
                + rng.normal(loc=0.0, scale=0.008)
            )

        profiles[:, client] = np.clip(
            base_level + daily_cycle + slow_cycle + jitter,
            0.65,
            1.30,
        )

    return profiles


def generate_bus_loads(
    number_of_buses: int,
    number_of_steps: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    utility_index = assign_buses_to_utilities(number_of_buses)
    number_of_clients = utility_index.max() + 1

    profiles = generate_utility_profiles(
        number_of_steps=number_of_steps,
        number_of_clients=number_of_clients,
        seed=seed,
    )

    # Synthetic base load for each bus.
    base_p_mw = rng.uniform(10.0, 150.0, size=number_of_buses)

    # Reactive load is approximately 20% to 40% of real load.
    reactive_ratio = rng.uniform(0.20, 0.40, size=number_of_buses)

    timestamps = pd.date_range(
        start="2026-01-01",
        periods=number_of_steps,
        freq="1min",
    )

    records = []

    for step, timestamp in enumerate(timestamps):
        for bus_index in range(number_of_buses):
            client_index = utility_index[bus_index]

            # Small bus-specific variation prevents buses within the same
            # utility from having perfectly identical curves.
            local_variation = rng.normal(loc=1.0, scale=0.01)

            p_mw = (
                base_p_mw[bus_index]
                * profiles[step, client_index]
                * local_variation
            )
            q_mvar = p_mw * reactive_ratio[bus_index]

            records.append(
                {
                    "timestamp": timestamp,
                    "bus_id": bus_index + 1,
                    "utility_id": client_index,
                    "utility_name": NYISO_CLIENTS[client_index],
                    "p_mw": p_mw,
                    "q_mvar": q_mvar,
                    "is_attack": 0,
                    "attack_type": "none",
                }
            )

    bus_data = pd.DataFrame(records)

    utility_data = (
        bus_data.groupby(
            ["timestamp", "utility_id", "utility_name"],
            as_index=False,
        )[["p_mw", "q_mvar"]]
        .sum()
        .sort_values(["timestamp", "utility_id"])
    )

    return bus_data, utility_data


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--buses",
        type=int,
        default=14,
        help="Number of buses in the test system",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1440,
        help="Timeline length in minutes",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for repeatable results",
    )

    args = parser.parse_args()

    output_directory = Path("generated_data")
    output_directory.mkdir(exist_ok=True)

    bus_data, utility_data = generate_bus_loads(
        number_of_buses=args.buses,
        number_of_steps=args.steps,
        seed=args.seed,
    )

    bus_path = output_directory / f"bus_loads_{args.buses}.csv"
    utility_path = output_directory / f"utility_loads_{args.buses}.csv"

    bus_data.to_csv(bus_path, index=False)
    utility_data.to_csv(utility_path, index=False)

    print(f"Generated {len(bus_data):,} bus measurements")
    print(f"Generated {len(utility_data):,} utility measurements")
    print(f"Saved: {bus_path}")
    print(f"Saved: {utility_path}")
    print()
    print(bus_data.head())


if __name__ == "__main__":
    main()

    