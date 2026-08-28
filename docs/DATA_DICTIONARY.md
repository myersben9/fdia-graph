# Data dictionary

What every array from `fg.load()` / `fg.load_stream()` holds.

**N** = number of buses (nodes), **E** = number of branches (edges). A shape is "values per item":
`[N,4]` = 4 numbers per bus, `[E,8]` = an 8-dim vector per branch, `[2,E]` = 2 rows × E branches.

## Cheat sheet

```
node_x [N,4]    = [ |V| , P_inj , Q_inj , theta ]     bus meter readings
node_m [N,4]    = 1 metered / 0 not (value zero-filled)
edge_x [E,2]    = [ P_from , Q_from ]                 power leaving branch (sign = direction)
edge_m [E,2]    = 1 metered / 0 not
edge_index [2,E]= [ from_bus ; to_bus ]               connectivity
edge_attr  [E,8]= [ r, x, b, g, gs, bs, tap, shift ]  static branch electrical properties
y [N]           = 1 attacked / 0 clean                localization target
swing [N,2]     = z-scored scan-to-scan change on [P,Q]
clean [N,4]     = [ |V| , P_inj , Q_inj , theta ]     noiseless truth, ALL buses (SE target)
edge_clean [E,2]= [ P_from , Q_from ]                 noiseless true flows (metered branches)
```

## `edge_attr` `[E,8]` — static branch properties (per-unit, never change)

| col | name | meaning |
|-----|------|---------|
| 0 | `r` | series **impedance**, real part (resistance) — `Z = r + jx` |
| 1 | `x` | series impedance, imag part (reactance) |
| 2 | `b` | line-charging susceptance (shunt) |
| 3 | `g` | shunt conductance (usually 0) |
| 4 | `gs` | series **admittance**, real part (conductance) — `Y = 1/(r+jx) = gs + j·bs` |
| 5 | `bs` | series admittance, imag part (susceptance) |
| 6 | `tap` | transformer tap ratio (1.0 = plain line) |
| 7 | `shift` | transformer phase shift, degrees (0 = plain line) |

`r,x` (impedance) and `gs,bs` (admittance) are the same branch inverted (`Y = 1/Z`) — both included so you
don't have to compute one from the other.

Note: in `format="pyg"`, `Data.edge_attr` is the `[E,2]` flows, not this. The `[E,8]` is `ds.edge_attr`.

## `node_x` `[N,4]` — bus measurements (voltage first)

| col | name | physical units | pu units |
|-----|------|----------------|----------|
| 0 | `|V|` | per-unit | per-unit |
| 1 | `P_inj` | MW | pu |
| 2 | `Q_inj` | MVAr | pu |
| 3 | `theta` | degrees | radians |

`P_inj`/`Q_inj` sign: `+` = net consumption (load), `−` = net injection (gen) — pandapower's `res_bus`
convention. Example (case14): the slack bus reads ≈ −235 MW, a 94 MW load bus reads ≈ +93 MW.

## Others

- `edge_x` `[E,2]` = `[P_from, Q_from]`, power leaving the from-end (sign = direction). MW/MVAr or pu.
- `node_m` / `edge_m`: `1` metered, `0` not. Metering is sparse; read the mask.
- `edge_index` `[2,E]`: row 0 from-bus, row 1 to-bus.
- `y` `[N]`: per-bus attack label. `family`: 0 benign, 1 Aq, 2 Ad, 3 As, 4 Ar, 5 At, 6 Al (Aq = paper `A_o`).
- `stealthy`, `split` (0/1/2 = train/val/test), `timestep`.
- `temporal_delta` `[N,2]` = scan-to-scan `[ΔP, ΔQ]`; `swing` `[N,2]` = that as a z-score of recent volatility.
- `clean` `[N,4]` / `edge_clean` `[E,2]` (v0.7.2+): the NOISELESS attack-free truth at the record's
  timestep, in `node_x` column order `[|V|, P_inj, Q_inj, theta]` — the SE / reconstruction target,
  same layer the streams ship. `clean` is full (every bus, no mask, regardless of meter placement);
  `edge_clean` zeroes unmetered branches like `edge_x`. On benign records `node_x − clean` = meter noise.

## Streams (`fg.load_stream`)

Leading time axis `T`, three aligned layers each for node and edge:

| layer | meaning |
|-------|---------|
| `node_x` / `edge_x` | observed (attacked + noise) — model input |
| `benign` / `edge_benign` | attack removed, noise kept |
| `clean` / `edge_clean` | noiseless true state — the SE target |

`benign − clean` = noise. `observed − benign` = the attack (exact for Ad/As/Ar; carries a noise term for
Aq/At/Al, so use `clean` as the SE target there).
