"""Run AC power flow for every scaled-load snapshot."""

from copy import deepcopy
from pathlib import Path
import argparse
import importlib.util

import numpy as np
import pandas as pd

from pypower.api import ppoption, runpf
from pypower.idx_bus import BUS_I, PD, QD, VM, VA
from pypower.idx_brch import (
    F_BUS,
    T_BUS,
    RATE_A,
    PF,
    QF,
    PT,
    QT,
)


def load_power_case(case_path: str) -> dict:
    """Load a PYPOWER Python case file."""
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

    case_function_name = path.stem
    case_function = getattr(
        module,
        case_function_name,
        None,
    )

    if case_function is None:
        raise AttributeError(
            f"The file {path.name} does not contain "
            f"{case_function_name}()."
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


def validate_scaled_loads(
    power_case: dict,
    scaled_loads: pd.DataFrame,
) -> None:
    """Validate the Part 2 scaled-load file."""
    required_columns = {
        "snapshot",
        "timestamp",
        "bus_id",
        "p_mw",
        "q_mvar",
    }

    missing_columns = required_columns.difference(
        scaled_loads.columns
    )

    if missing_columns:
        raise ValueError(
            "The scaled-load file is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if scaled_loads.empty:
        raise ValueError("The scaled-load file is empty.")

    if scaled_loads[
        ["p_mw", "q_mvar"]
    ].isna().any().any():
        raise ValueError(
            "The scaled-load file contains missing values."
        )

    case_bus_ids = set(
        power_case["bus"][:, BUS_I].astype(int)
    )

    load_bus_ids = set(
        scaled_loads["bus_id"].astype(int).unique()
    )

    if case_bus_ids != load_bus_ids:
        raise ValueError(
            "The bus numbers in the scaled-load file do not "
            "match the power-system case."
        )


def apply_snapshot_loads(
    base_case: dict,
    snapshot_loads: pd.DataFrame,
) -> dict:
    """Apply one snapshot's P and Q loads to a case copy."""
    snapshot_case = deepcopy(base_case)

    bus_row_by_id = {
        int(bus_id): row_index
        for row_index, bus_id in enumerate(
            snapshot_case["bus"][:, BUS_I]
        )
    }

    for record in snapshot_loads.itertuples(
        index=False
    ):
        bus_id = int(record.bus_id)

        if bus_id not in bus_row_by_id:
            raise ValueError(
                f"Bus {bus_id} was not found in the case."
            )

        row_index = bus_row_by_id[bus_id]

        snapshot_case["bus"][row_index, PD] = (
            float(record.p_mw)
        )
        snapshot_case["bus"][row_index, QD] = (
            float(record.q_mvar)
        )

    return snapshot_case


def collect_bus_results(
    solved_case: dict,
    snapshot: int,
    timestamp: str,
) -> list[dict]:
    """Collect solved bus voltages and loads."""
    records = []

    for bus_row in solved_case["bus"]:
        records.append(
            {
                "snapshot": snapshot,
                "timestamp": timestamp,
                "bus_id": int(bus_row[BUS_I]),
                "p_mw": float(bus_row[PD]),
                "q_mvar": float(bus_row[QD]),
                "voltage_pu": float(bus_row[VM]),
                "voltage_angle_deg": float(bus_row[VA]),
            }
        )

    return records


def collect_branch_results(
    solved_case: dict,
    snapshot: int,
    timestamp: str,
) -> list[dict]:
    """Collect solved branch power flows."""
    records = []

    for branch_index, branch_row in enumerate(
        solved_case["branch"],
        start=1,
    ):
        p_from = float(branch_row[PF])
        q_from = float(branch_row[QF])
        p_to = float(branch_row[PT])
        q_to = float(branch_row[QT])
        rate_a = float(branch_row[RATE_A])

        apparent_power_from = np.hypot(
            p_from,
            q_from,
        )
        apparent_power_to = np.hypot(
            p_to,
            q_to,
        )

        maximum_apparent_power = max(
            apparent_power_from,
            apparent_power_to,
        )

        if rate_a > 0:
            loading_percent = (
                100.0
                * maximum_apparent_power
                / rate_a
            )
        else:
            loading_percent = np.nan

        records.append(
            {
                "snapshot": snapshot,
                "timestamp": timestamp,
                "branch_id": branch_index,
                "from_bus": int(branch_row[F_BUS]),
                "to_bus": int(branch_row[T_BUS]),
                "p_from_mw": p_from,
                "q_from_mvar": q_from,
                "p_to_mw": p_to,
                "q_to_mvar": q_to,
                "rate_a_mva": rate_a,
                "loading_percent": loading_percent,
            }
        )

    return records


def run_timeline_powerflow(
    base_case: dict,
    scaled_loads: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run AC power flow for every timeline snapshot."""
    powerflow_options = ppoption(
        PF_ALG=1,
        VERBOSE=0,
        OUT_ALL=0,
    )

    bus_records = []
    branch_records = []
    convergence_records = []

    grouped_snapshots = scaled_loads.groupby(
        "snapshot",
        sort=True,
    )

    total_snapshots = scaled_loads[
        "snapshot"
    ].nunique()

    for completed, (
        snapshot,
        snapshot_loads,
    ) in enumerate(grouped_snapshots, start=1):
        snapshot = int(snapshot)
        timestamp = str(
            snapshot_loads["timestamp"].iloc[0]
        )

        snapshot_case = apply_snapshot_loads(
            base_case=base_case,
            snapshot_loads=snapshot_loads,
        )

        solved_case, success = runpf(
            snapshot_case,
            powerflow_options,
        )

        converged = bool(success)

        convergence_records.append(
            {
                "snapshot": snapshot,
                "timestamp": timestamp,
                "converged": converged,
            }
        )

        if converged:
            bus_records.extend(
                collect_bus_results(
                    solved_case=solved_case,
                    snapshot=snapshot,
                    timestamp=timestamp,
                )
            )

            branch_records.extend(
                collect_branch_results(
                    solved_case=solved_case,
                    snapshot=snapshot,
                    timestamp=timestamp,
                )
            )
        else:
            print(
                f"Warning: snapshot {snapshot} "
                "did not converge."
            )

        if (
            completed == 1
            or completed % 25 == 0
            or completed == total_snapshots
        ):
            print(
                f"Completed {completed} "
                f"of {total_snapshots} snapshots"
            )

    return (
        pd.DataFrame(bus_records),
        pd.DataFrame(branch_records),
        pd.DataFrame(convergence_records),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run AC power flow for every snapshot "
            "in a scaled-load timeline."
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
        help="Path to the Part 2 scaled-load CSV",
    )

    parser.add_argument(
        "--output-directory",
        required=True,
        help="Directory for the power-flow results",
    )

    args = parser.parse_args()

    scaled_load_path = Path(args.loads)

    if not scaled_load_path.exists():
        raise FileNotFoundError(
            "The Part 2 scaled-load file was not found: "
            f"{scaled_load_path.resolve()}"
        )

    print("Loading the power-system case...")
    base_case = load_power_case(args.case)

    print("Loading the Part 2 scaled loads...")
    scaled_loads = pd.read_csv(scaled_load_path)

    validate_scaled_loads(
        power_case=base_case,
        scaled_loads=scaled_loads,
    )

    print("Running the AC power-flow timeline...")
    print()

    (
        bus_results,
        branch_results,
        convergence_results,
    ) = run_timeline_powerflow(
        base_case=base_case,
        scaled_loads=scaled_loads,
    )

    output_directory = Path(args.output_directory)
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    bus_output = output_directory / "bus_results.csv"
    branch_output = (
        output_directory / "branch_results.csv"
    )
    convergence_output = (
        output_directory / "convergence.csv"
    )

    bus_results.to_csv(
        bus_output,
        index=False,
    )
    branch_results.to_csv(
        branch_output,
        index=False,
    )
    convergence_results.to_csv(
        convergence_output,
        index=False,
    )

    converged_count = int(
        convergence_results["converged"].sum()
    )
    total_count = len(convergence_results)

    print()
    print("Timeline AC power flow complete")
    print(
        f"Converged snapshots: "
        f"{converged_count}/{total_count}"
    )
    print(f"Bus results: {bus_output}")
    print(f"Branch results: {branch_output}")
    print(
        f"Convergence results: "
        f"{convergence_output}"
    )

    if converged_count != total_count:
        print()
        print(
            "Warning: at least one snapshot failed. "
            "Check convergence.csv."
        )


if __name__ == "__main__":
    main()