# Data dictionary — what every number means

New here? This page says, in plain English, what each array you get from `fg.load()` /
`fg.load_stream()` contains, column by column, with units. Keep it open next to your code.

The grid has **N buses** (nodes) and **E branches** (lines + transformers). Everything is either a
**per-bus** array (first feature axis `N`), a **per-branch** array (axis `E`), or a **per-record** scalar.

---

## The one you keep asking about: `edge_attr` `[E, 8]`

These are the **static electrical properties of each branch** (line or transformer). They never change
between records — they describe the wire, not the power flowing through it. All per-unit on the system
base (100 MVA).

| col | name    | plain meaning |
|-----|---------|---------------|
| 0   | `r`     | series **resistance** of the branch (opposes current, causes I²R loss) |
| 1   | `x`     | series **reactance** (the inductive part; dominates power flow on transmission lines) |
| 2   | `b`     | line-charging **susceptance** (the shunt capacitance of the line to ground) |
| 3   | `g`     | shunt **conductance** (usually 0 for lines) |
| 4   | `gs`    | real part of the **series admittance** `1/(r + jx)` |
| 5   | `bs`    | imaginary part of the series admittance `1/(r + jx)` (negative for an inductive branch) |
| 6   | `tap`   | transformer **tap ratio** (1.0 for a plain line; ≠1 steps voltage up/down) |
| 7   | `shift` | transformer **phase-shift angle** in degrees (0 for a plain line) |

Together, `r, x, b, g, tap, shift` are exactly what you need to build the bus admittance matrix **Ybus**.
`gs, bs` are just `1/(r+jx)` precomputed so you don't have to. Example (a real 118-bus line):
`[0.0303, 0.0999, 0.0254, 0.0, 2.78, -9.17, 1.0, 0.0]` → `r=0.03, x=0.10 pu`, some line charging,
series admittance `2.78 − j9.17`, a plain line (tap 1, no shift).

> Careful: in the **PyG format** (`format="pyg"`), `Data.edge_attr` is **not** this `[E,8]` — there it is
> the `[E,2]` branch-flow *measurements* (see `edge_x` below). The static `[E,8]` line features are on the
> loader object as `ds.edge_attr`, and in the dict format under the key `edge_attr`.

---

## Per-bus measurements: `node_x` `[N, 4]`

What the meters read at each bus. Column order is **`[|V|, P_inj, Q_inj, θ]`** (voltage first — the
column that sits near 1.0 is the voltage magnitude, not power).

| col | name    | plain meaning | units (`units="physical"`, default) | units (`units="pu"`) |
|-----|---------|---------------|--------------------------------------|----------------------|
| 0   | `|V|`   | bus **voltage magnitude** | per-unit (≈ 0.9–1.1) | per-unit |
| 1   | `P_inj` | net **active power injection** (generation − load) | MW | per-unit on 100 MVA |
| 2   | `Q_inj` | net **reactive power injection** | MVAr | per-unit |
| 3   | `θ`     | bus **voltage angle** | degrees | radians |

Sign of `P_inj`/`Q_inj`: positive = the bus is a net **source** (generator), negative = net **load**.

## Per-bus mask: `node_m` `[N, 4]`

Same shape as `node_x`. `1` = that channel is **metered** at that bus; `0` = **not observed** (and the
value in `node_x` is zero-filled). Metering is sparse, so a model must read the mask to tell a real `0`
reading from an unobserved channel.

## Per-branch measurements: `edge_x` `[E, 2]`

The **power flow leaving the "from" end** of each branch — `[P_from, Q_from]`. The **sign gives the
direction** (positive = power flows from→to). Units: MW/MVAr (`physical`) or per-unit (`pu`). Line
loading % ≈ `sqrt(P_from² + Q_from²) / branch_MVA_rating`.

## Per-branch mask: `edge_m` `[E, 2]`  — same convention as `node_m`.

## Connectivity: `edge_index` `[2, E]`

Row 0 = **from-bus** index, row 1 = **to-bus** index, for each branch (lines first, then transformers).
This is the COO format PyTorch-Geometric expects. Pairs with `edge_attr`/`edge_x` column-for-column.

---

## Labels and metadata (per record)

| field      | shape   | meaning |
|------------|---------|---------|
| `y`        | `[N]`   | per-bus **attack label**: `1` = this bus was attacked, `0` = clean. The localization target. |
| `family`   | scalar  | which attack: `0` benign, `1` Aq, `2` Ad, `3` As, `4` Ar, `5` At, `6` Al (Aq = the paper's `A_o`). |
| `stealthy` | scalar  | `1` if the attack passes bad-data detection + load-plausibility. |
| `split`    | scalar  | `0` train, `1` val, `2` test (chronological 60/20/20). |
| `timestep` | scalar  | index of the operating point this record came from. |

## Engineered temporal features (per bus)

| field            | shape   | meaning |
|------------------|---------|---------|
| `temporal_delta` | `[N,2]` | this scan minus the previous scan, on `[P, Q]` — the raw scan-to-scan change. |
| `swing`          | `[N,2]` | that change as a **z-score** of the bus's own recent volatility. A spike reads large, a slow ramp stays ≈ small. This is the feature that carries localization (see `CONCEPTS_TO_CODE.md`). |

---

## Continuous streams (`fg.load_stream`)

A stream is one running timeline of length `T`, with the same fields plus a leading time axis, and **three
aligned measurement layers** for both node and branch measurements:

| layer        | node shape   | edge shape   | meaning |
|--------------|--------------|--------------|---------|
| `node_x` / `edge_x`           | `[T,N,4]` / `[T,E,2]` | **observed** — attacked + meter noise (the model input) |
| `benign` / `edge_benign`      | `[T,N,4]` / `[T,E,2]` | attack **removed**, meter noise kept |
| `clean` / `edge_clean`        | `[T,N,4]` / `[T,E,2]` | **noiseless, attack-free true state** — the state-estimation target |

`benign − clean` is meter noise. `observed − benign` isolates the attack **exactly** for the meter-tamper
families (Ad/As/Ar); for the re-solve families (Aq/At/Al) it also carries a small noise difference, so use
`clean` as the SE target there.

---

## Two-second cheat sheet

```
node_x [N,4]  = [ |V| ,  P_inj ,  Q_inj ,  theta ]      what the bus meters read
node_m [N,4]  = 1 where metered, else 0
edge_x [E,2]  = [ P_from , Q_from ]                     power leaving the branch (sign = direction)
edge_m [E,2]  = 1 where metered, else 0
edge_index [2,E] = [ from_bus ; to_bus ]               connectivity
edge_attr  [E,8] = [ r, x, b, g, gs, bs, tap, shift ]  static branch electrical properties
y [N]         = 1 where attacked                       localization target
swing [N,2]   = z-scored scan-to-scan change on [P,Q]  the "did this bus just jump" feature
```
