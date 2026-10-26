"""Scale base P and Q bus loads using a generated load series."""

from pathlib import Path
import argparse
import importlib.util

import numpy as np
import pandas as pd

from pypower.idx_bus import BUS_I, PD, QD


def load_power_case(case_path: str) -> dict:
    """
    Load a PYPOWER Python case file.

    This works with case14.py and future PYPOWER case files.
    """
    path = Path(case_path).resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Power-system case was not found: {path}"
        )

    if path.suffix.lower() != ".py":
        raise ValueError(
            "The power-system case must currently be a .py file."
        )

    spec = importlib.util.spec_from_file_location(
        "selected_power_case",
        path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not import the case file: {path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # A file named case14.py normally contains case14().
    case_function_name = path.stem
    case_function = getattr(
        module,
        case_function_name,
        None,
    )

    if case_function is None:
        raise AttributeError(
            f"The file {path.name} does not contain a function "
            f"named {case_function_name}()."
        )

    power_case = case_function()

    if not isinstance(power_case, dict):
        raise ValueError(
            f"{case_function_name}() did not return "
            "a PYPOWER case dictionary."
        )

    required_keys = {"baseMVA", "bus", "gen", "branch"}
    missing_keys = required_keys.difference(power_case)

    if missing_keys:
        raise ValueError(
            "The case is missing required values: "
            f"{sorted(missing_keys)}"
        )

    return power_case


def validate_load_series(load_series: pd.DataFrame) -> None:
    """Check that the Part 1 file contains the required columns."""
    required_columns = {
        "snapshot",
        "timestamp",
        "bus_id",
        "utility_id",
        "utility_name",
        "system_factor",
        "bus_fluctuation",
        "load_multiplier",
    }

    missing_columns = required_columns.difference(
        load_series.columns
    )

    if missing_columns:
        raise ValueError(
            "The load-series file is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if load_series.empty:
        raise ValueError("The load-series file is empty.")

    if load_series["load_multiplier"].isna().any():
        raise ValueError(
            "The load-series file contains missing multipliers."
        )

    if (load_series["load_multiplier"] <= 0).any():
        raise ValueError(
            "Every load multiplier must be greater than zero."
        )


def scale_bus_loads(
    power_case: dict,
    load_series: pd.DataFrame,
) -> pd.DataFrame:
    """
    Scale the original case loads at every snapshot.

    P_i,t = P_i,base * load_multiplier_i,t
    Q_i,t = Q_i,base * load_multiplier_i,t
    """
    bus = power_case["bus"]

    actual_bus_ids = bus[:, BUS_I].astype(int)

    generated_bus_ids = np.sort(
        load_series["bus_id"].unique().astype(int)
    )

    if len(actual_bus_ids) != len(generated_bus_ids):
        raise ValueError(
            "Bus-count mismatch: "
            f"the power case contains {len(actual_bus_ids)} buses, "
            f"but the generated series contains "
            f"{len(generated_bus_ids)} buses."
        )

    if set(actual_bus_ids) != set(generated_bus_ids):
        raise ValueError(
            "The bus numbers in the generated load series do not "
            "match the bus numbers in the power-system case."
        )

    # Store each bus's original real and reactive load.
    base_loads = pd.DataFrame(
        {
            "bus_id": actual_bus_ids,
            "base_p_mw": bus[:, PD],
            "base_q_mvar": bus[:, QD],
        }
    )

    # Join the Part 1 multipliers to the actual case loads.
    scaled_loads = load_series.merge(
        base_loads,
        on="bus_id",
        how="left",
        validate="many_to_one",
    )

    if scaled_loads[
        ["base_p_mw", "base_q_mvar"]
    ].isna().any().any():
        raise ValueError(
            "Some generated bus numbers could not be matched "
            "to the power-system case."
        )

    # Assignment Part 2 calculations.
    scaled_loads["p_mw"] = (
        scaled_loads["base_p_mw"]
        * scaled_loads["load_multiplier"]
    )

    scaled_loads["q_mvar"] = (
        scaled_loads["base_q_mvar"]
        * scaled_loads["load_multiplier"]
    )

    output_columns = [
        "snapshot",
        "timestamp",
        "bus_id",
        "utility_id",
        "utility_name",
        "system_factor",
        "bus_fluctuation",
        "load_multiplier",
        "base_p_mw",
        "base_q_mvar",
        "p_mw",
        "q_mvar",
    ]

    return (
        scaled_loads[output_columns]
        .sort_values(["snapshot", "bus_id"])
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scale the base P and Q loads in a PYPOWER case "
            "for every generated snapshot."
        )
    )

    parser.add_argument(
        "--case",
        required=True,
        help="Path to a PYPOWER Python case file",
    )

    parser.add_argument(
        "--loads",
        required=True,
        help="Path to the Part 1 load-series CSV",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path for the scaled-load output CSV",
    )

    args = parser.parse_args()

    load_series_path = Path(args.loads)

    if not load_series_path.exists():
        raise FileNotFoundError(
            "The generated load-series file was not found: "
            f"{load_series_path.resolve()}"
        )

    print("Loading the power-system case...")
    power_case = load_power_case(args.case)

    print("Loading the Part 1 load series...")
    load_series = pd.read_csv(load_series_path)

    validate_load_series(load_series)

    print("Scaling the base bus loads...")
    scaled_loads = scale_bus_loads(
        power_case=power_case,
        load_series=load_series,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    scaled_loads.to_csv(
        output_path,
        index=False,
    )

    number_of_buses = scaled_loads["bus_id"].nunique()
    number_of_snapshots = (
        scaled_loads["snapshot"].nunique()
    )

    print()
    print("Bus-load scaling complete")
    print(f"Buses: {number_of_buses}")
    print(f"Snapshots: {number_of_snapshots}")
    print(f"Rows: {len(scaled_loads):,}")
    print(f"Saved to: {output_path}")
    print()
    print("First 10 scaled loads:")
    print(scaled_loads.head(10))
    print()
    print("Scaled real-power summary:")
    print(scaled_loads["p_mw"].describe())


if __name__ == "__main__":
    main()