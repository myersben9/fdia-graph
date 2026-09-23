# Localization with `fdia_graph.localization`

Which buses are under attack, per scan.

```python
import fdia_graph as fg
from fdia_graph.localization import SwingThreshold, DeltaThreshold, ResidualLocalizer, BusCNN, BusMLP

train = fg.load("ieee14", split="train")
test  = fg.load("ieee14", split="test")

loc  = SwingThreshold(fa_target=0.01).fit(train)   # thresholds set on benign records only
flag = loc.localize(test)                          # [n, N] bool: which buses are called attacked
rep  = loc.score(test)                             # per-family metrics + benign false alarms
```

![loc.fit calibrates threshold arms on benign records and trains the learned arms on every record; loc.localize flags buses above the per-bus threshold; loc.score reports per-family metrics](../figures/diagrams/localization_flow.png)

| | |
|---|---|
| shared | metrics and the benign-quantile calibration in `LocalizerBase`; a threshold arm changes only the per-bus score, a learned arm also overrides `fit` to train on every record and, given `val`, the threshold |
| budget | every method runs at the same false-alarm rate, set on benign records; the one exception is a learned arm given `val`, which picks one global validation-best threshold from labelled records (the papers' zero-shot protocol) |
| `"all"` entry | pools every record; the papers' per-bus macro F1, DR and FR over the attackable buses |

## The methods

| Class | Score | Needs |
|---|---|---|
| `SwingThreshold` | The dataset's windowed swing feature: each scan's injection change as a z-score of the bus's typical recent change | numpy only |
| `DeltaThreshold` | The raw one-scan change scaled by the bus's benign RMS. Same signal without the windowing, so the gap shows what windowing buys | numpy only |
| `ResidualLocalizer` | Largest normalized residual from a state-estimation solve (any `fdia_graph.se` estimator, default `WLS`), aggregated to each bus's own meters and incident flows. Textbook bad-data identification | `[se]` extra |
| `BusMLP` | The papers' lightweight arm: one 4x128 MLP applied to every bus's own 14-dim vector (readings, meter mask, partial KCL residual, delta, swing). 52k parameters | `[torch]` extra |
| `BusCNN` | The papers' best localizer: a 1-D convolution across the bus axis over the same 14-dim vector, 4 layers of 128 channels, kernel 3. No graph read. 154k parameters | `[torch]` extra |

The learned arms train on the records they are given, so the protocol is the load call;
`fit(train, val=val)` picks the papers' validation-best threshold instead of the false-alarm one.

## Results

| protocol | train and val | test | threshold |
|---|---|---|---|
| zero-shot (the paper's) | benign + `Aq` + `Ad` | adds `As` and `Ar`, never seen in training | learned arms validation-best; swing keeps its benign calibration |
| common | every family, unfiltered | the full split | benign quantile at `fa_target=0.01` for every method |

F1, DR and FR are the paper's per-bus macro scores over the attackable buses (F1 and recall over
every test record, FR the per-bus false-positive rate on benign records); full metrics in
`results/loc_ieee{14,118,300}.json`.

**Zero-shot protocol**

| Method | F1 14 | DR 14 | FR 14 | F1 118 | DR 118 | FR 118 | F1 300 | DR 300 | FR 300 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Per-bus MLP | 0.7443 | 0.6561 | 0.0062 | 0.5343 | 0.4164 | 0.0016 | 0.5009 | 0.4615 | 0.0024 |
| **1D CNN** | 0.7455 | 0.6438 | 0.0001 | 0.5473 | 0.4371 | 0.0010 | 0.5322 | 0.4961 | 0.0024 |
| Swing threshold | 0.1245 | 0.0698 | 0.0069 | 0.5372 | 0.6093 | 0.0098 | 0.5153 | 0.7934 | 0.0091 |

The paper reports 0.9634 / 0.9625 / 0.9524 for the CNN and 0.9626 / 0.9570 / 0.9327 for the MLP on
v0.4.1 data, and the SDK classes reproduce them on the v0.7.2 record shards. The v0.8.0 timeline columns are lower for the same detectors because their temporal features compare each frame with the frame emitted one minute earlier rather than with the benign scan before an attacked snapshot, so a sustained episode spikes at its onset and a one-frame family also spikes on the benign frame after it. Those post-attack benign frames stay in the benign calibration set on purpose: an operator's detector sees them too, so leaving them out would set lower thresholds and let the realized false-alarm rate exceed `fa_target`.

**Common protocol**

| Method | F1 14 | DR 14 | FR 14 | F1 118 | DR 118 | FR 118 | F1 300 | DR 300 | FR 300 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Swing threshold | 0.0778 | 0.0417 | 0.0069 | 0.4044 | 0.4035 | 0.0098 | 0.4310 | 0.6399 | 0.0091 |
| Delta threshold | 0.0818 | 0.0439 | 0.0066 | 0.4122 | 0.4212 | 0.0100 | 0.4325 | 0.6419 | 0.0091 |
| Residual (LNR) | 0.3433 | 0.3083 | 0.0027 | 0.1983 | 0.3810 | 0.0063 | 0.1953 | 0.5707 | 0.0050 |
| Per-bus MLP | 0.7152 | 0.6190 | 0.0112 | 0.4821 | 0.5190 | 0.0103 | 0.4355 | 0.6634 | 0.0102 |
| **1D CNN** | 0.8038 | 0.8601 | 0.0146 | 0.4978 | 0.6255 | 0.0110 | 0.4033 | 0.7050 | 0.0111 |
| 1D CNN + Jacobian (C) | 0.8699 | 0.9730 | 0.0151 | 0.4670 | 0.9114 | 0.0097 | 0.2866 | 0.9087 | 0.0093 |

Per-bus F1 by attack family, common protocol. Row labels carry each method's FR.

| IEEE 14 | IEEE 118 | IEEE 300 |
|---|---|---|
| ![](results/fig_loc_ieee14.png) | ![](results/fig_loc_ieee118.png) | ![](results/fig_loc_ieee300.png) |

## Jacobian-informed features: the digest's ablation

`fdia_graph.se.JacobianFeatures` transforms the measurement change through the estimator's Jacobian
(implied state move `H⁺Δz`, explained and unexplained parts and their ratio, sensitivity, leverage,
weak-direction energy) and aggregates it to buses; `BusCNN` / `BusMLP` take `features=`:

| set | input |
|---|---|
| A | measurements only |
| B | the papers' 14-dim vector (the rows above) |
| C | B + the 8 Jacobian features |
| D | the Jacobian features alone |

Zero-shot protocol on the v0.8.0 timelines (benign, `Aq` and `Ad` seen; `As` and `Ar` unseen):

| Model (1D CNN, zero-shot) | F1 14 | DR 14 | FR 14 | F1 118 | DR 118 | FR 118 | F1 300 | DR 300 | FR 300 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A: measurements only | 0.606 | 0.641 | 0.0001 | 0.000 | 0.000 | 0.0000 | 0.005 | 0.003 | 0.0001 |
| B: measurements + temporal (the papers' 14) | 0.745 | 0.644 | 0.0001 | 0.547 | 0.437 | 0.0010 | 0.532 | 0.496 | 0.0024 |
| C: B + Jacobian features | **0.865** | 0.812 | 0.0000 | **0.859** | 0.790 | 0.0000 | **0.848** | 0.781 | 0.0000 |
| D: Jacobian features only | 0.795 | 0.737 | 0.0000 | 0.779 | 0.711 | 0.0000 | 0.750 | 0.703 | 0.0000 |

In the common protocol (every family in distribution) C wins on 14 (0.870 vs 0.804) and loses on 118
(0.467 vs 0.498) and 300 (0.287 vs 0.403) at the same benign alarm rate: it detects far more frames
(0.91 against 0.63 and 0.71) and pays in precision on the buses around a local false state.

| finding | evidence |
|---|---|
| the features carry the digest's signal | on `Aq` the zero-shot node-F1 is 0.65, 0.04 and 0.06 for B on 14, 118 and 300 and 0.86, 0.77 and 0.50 for D alone: the per-frame physics sees the local false state where the history does not |
| on a timeline they are what makes the vector work | B to C is +12, +31 and +32 zero-shot points; the papers' vector was built for the v0.7.2 shards, whose temporal features compared an attacked snapshot with the benign scan before it, and on a timeline a sustained episode spikes at its onset only |
| where they pay again | as the localizer that gates the estimator once meters are secured, [`../trust/README.md`](../trust/README.md) |

## Three readings

| reading | evidence | open case |
|---|---|---|
| the temporal spike is an onset signal | the swing threshold alone reads 0.08, 0.40 and 0.43 macro-F1 in the common protocol on 14, 118 and 300: it catches the one-frame families and the first frame of an episode, then the feature fades because each frame is compared with the frame emitted a minute earlier | the slow ramp `At` and the held redistribution `Am` inside an episode |
| the classical arm misses every stealthy family | `ResidualLocalizer` finds in-place corruption and smears it over neighbours; on `Aq` / `At` / `Al` / `Am` its node-F1 is 0.01 or less, there is no residual | it opens with a trusted set of meters, [`../trust/README.md`](../trust/README.md) |
| learning plus physics holds with size | zero-shot CNN with the Jacobian block 0.865, 0.859 and 0.848 from 14 to 300 buses at FR 1e-4 or below | the common protocol, with `At`, `Al` and `Am` in distribution, falls 0.870, 0.467, 0.287 with size: the per-frame localization of a sustained local false state is the frontier |

## Regenerate

```bash
FG_SYSTEM=ieee14 python docs/localization/run_localization.py    # fits, writes results/loc_ieee14.json
FG_SYSTEM=ieee118 python docs/localization/run_localization.py
FG_SYSTEM=ieee300 python docs/localization/run_localization.py
python docs/localization/make_report.py                          # tables (markdown) + figures + CSV from the JSON
```

Minutes for IEEE-14 on a GPU laptop; about half an hour for 300, mostly the residual arm's solves.
