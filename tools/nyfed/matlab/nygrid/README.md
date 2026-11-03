# NYgrid AC feasibility (MATLAB reference)

NYgrid's published 2019 operating points are DC. In AC none of the 8,760 hours converge. These scripts
are the MATLAB reference for the changes that make them AC-feasible, and for the test of which of those
changes are needed. The Python port in `src/nyfed/grid/fixes.py` and `scenario/solver.py` is checked
against the files they write.

## Setup

1. NYgrid checkout at commit 47698b6 in `data/NYgrid` and MATPOWER 8.0 in `data/matpower8.0`
   (both git-ignored). MATLAB R2025a was used.
2. Apply the one-line patch, so `updateOpCond` also returns the full 140-bus case it builds before the
   network reduction, and create the output folder NYgrid writes to:

   ```bash
   cd data/NYgrid
   git apply --ignore-whitespace ../../tools/matlab/nygrid/nygrid_full_case.patch
   mkdir -p Result
   ```

3. Run every script from `data/NYgrid`, since NYgrid reads `Data/` relative to the working folder.

## Files

| file | does |
|---|---|
| `fixes.m` | `'J'` splits the zone J units between NYC buses 81 and 82 by load, `'L'` restores the 765 kV Massena-Marcy line (47-43) left commented out in `modifyMPC.m` |
| `extbase.m` | holds the external areas at base-case load and scales the external units to cover NY's import |
| `acproj.m` | min-deviation AC-OPF, the AC-feasible dispatch closest to NYgrid's (minimizes sum (PG - PG_hist)^2) |
| `ac_feas.m` | first diagnosis: NYgrid's hours in DC, reduced-case AC and full-case AC, no changes |
| `ac_year.m` | the recipe (`fixes 'JL'` + `extbase` + `acproj`) over 2019, writes `results/nygrid.ac_year.csv` |
| `ablation.m` | drops one change at a time (`noJ`, `noL`, `noE`, `projOnly`), writes `results/ablation_parts/` |
| `dump_hours.m` | three operating points with the recipe, the fixtures in `tests/data/matlab_ref` |
| `nygrid_full_case.patch` | the change to NYgrid's `updateOpCond.m` |

## Running

One worker per MATLAB process, each taking every n-th hour:

```matlab
addpath('<nyfed>/tools/matlab/nygrid')
ablation('<nyfed>/data/matpower8.0', 1, 24, '<nyfed>/results/ablation_parts/part_1.csv')
```

`ac_year(mpdir, hours, outfile)` takes a datetime vector of hours instead. Then summarize:

```bash
python tools/nygrid_ac_summary.py year
python tools/nygrid_ac_summary.py ablation
```

The summaries are `results/nygrid.ac_year.json` and `results/nygrid.ac_ablation.json`.
