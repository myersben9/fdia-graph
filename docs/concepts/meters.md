# Meters and noise

A pool holds the true state of the grid at every frame. The operator does not see the state. The
operator sees meter readings, and each reading has an error. This page explains how the package
turns a state into a scan: the meter plan, the measurement function and the noise.

The three classes are in `src/fdia_graph/sensing/` and serve every test system.

| Class | File | Holds |
|---|---|---|
| `MeterPlan` | `sensing/meters.py` | every meter, as one row of a table |
| `Measurement` | `sensing/measure.py` | the readings a state gives, and their Jacobian |
| `NoiseModel` | `sensing/noise.py` | each meter's error: a bias and a jitter |

A benign scan is `NoiseModel.sample(Measurement.h(state))`. It needs no new solve: the readings
follow from the stored state.

## The meter plan

`MeterPlan` is a table with one row per meter.

| Column | Meaning |
|---|---|
| `kind` | what the meter reads (`MeterKind`, below) |
| `bus` | the bus it sits at |
| `branch`, `end` | the branch and its end (0 from, 1 to); -1 for a bus meter |
| `owner` | the client that owns the meter's bus |
| `sigma_rel`, `sigma_abs` | the accuracy class |
| `bias` | the meter's constant error |

The plan is made once per timeline, from the seed. It is the same on every frame.

### What a meter reads

The meters follow the SCADA and PMU plan of [WU26, eqs. 17-20]. A SCADA voltmeter reads the
voltage magnitude only. A PMU reads the magnitude, the angle, and the current phasor of every
branch end at its bus.

| `MeterKind` | Reads | Unit |
|---|---|---|
| `V` | voltage magnitude at a bus | p.u. |
| `VA` | voltage angle at a PMU bus | degrees |
| `P`, `Q` | power injection at a bus | MW, MVAr |
| `PF`, `QF` | power flow at one end of a branch | MW, MVAr |
| `IR`, `II` | PMU current as real and imaginary part | p.u. |

### The placement

`MeterPlan.from_coverage` places the meters at random, on every test system.

| Meter | Where |
|---|---|
| `P`, `Q` injection | every bus |
| `P`, `Q` flow | the from end of a random share of branches |
| Voltmeters | a random share of buses |
| PMUs | a random share of buses, at least one |
| PMU currents | every branch end at a PMU bus, as real and imaginary part |
| Owner | one client; with the owners of the buses, the client of the meter's bus |

`MeterSettings` gives the shares: `vbus_frac` 0.6, `pmu_frac` 0.2 and
`flow_frac` 0.9. The voltmeter buses are `int(0.6 N)` buses drawn at random, the PMU buses
`max(1, int(0.2 N))`. Each branch gets a flow meter with probability 0.9.

On the New York case the meters are drawn over all 140 buses, the neighbouring grids' included.
A source built with `own=` gives each meter the client of its bus; a meter at a bus that
belongs to no client has owner -1
([Generate a timeline](../guides/generation.md#a-timeline-read-as-clients)).

```python
import numpy as np

from fdia_graph.constants.meters import MeterKind
from fdia_graph.models.config import MeterSettings
from fdia_graph.sensing.meters import MeterPlan
from fdia_graph.systems.case import Case

case = Case.from_ieee(14)  # needs fdia-graph[generate]
rng = np.random.default_rng(0)
plan = MeterPlan.from_coverage(case, MeterSettings(), rng)
for kind in (MeterKind.V, MeterKind.VA, MeterKind.PF):
    print(kind.name, int((plan.kind == kind).sum()))
# V 10, VA 2, PF 17
```

### The table and the file

In the code the readings of one frame are a flat vector `z`, one value per row of the plan. In a
timeline file they are layers: one array per bus, per branch and per PMU current.

| Method | Does |
|---|---|
| `MeterPlan.scatter` | puts one value per meter at its place in the layers |
| `MeterPlan.gather` | reads the flat vector back from the layers |
| `MeterPlan.from_masks` | rebuilds the table from the meter masks of a file that records the client of each bus |

The round trip is exact. A client's data is the columns of `z` whose meter it owns.

## The measurement function

`Measurement.h(vm, va)` gives the exact reading of every meter of the plan for a state: the
voltage magnitudes `vm` (p.u.) and angles `va` (degrees). The flows and currents follow from the
voltages through the admittances of the `Network`.

The injections can be taken two ways.

- From the voltages. This is the default.
- From the pool, passed as `injections=`. An IEEE pool stores its own injections, and the
  generator passes them.

`Measurement.jacobian(vm, va)` is the derivative of the readings with respect to the state.
`Measurement.for_client(c)` is the same function over one client's meters.

```python
from fdia_graph.sensing.measure import Measurement
from fdia_graph.sensing.noise import NoiseModel
from fdia_graph.systems.network import Network

# case, plan and rng: from the snippet above
measure = Measurement(Network(case), plan)
z = measure.h(case.vm, case.va)       # exact readings
scan = NoiseModel(plan).sample(z, rng)  # one noisy scan
print(z.shape, scan.shape)            # (86,) (86,)
```

`Measurement.h` computes an injection from the voltages as generation minus load. A file stores
load as positive, the sign an IEEE pool uses ([Data dictionary](../reference/DATA_DICTIONARY.md)).

## Accuracy classes

Each meter kind has an accuracy class: the standard deviation of its total error. The classes
follow [ASP14] and are in `constants/meters.py` (`ACCURACY_CLASS`, `KIND_ACCURACY`).

| Meter | Standard deviation | Kind of error |
|---|---|---|
| power injection and flow | 1.7 % of the reading | relative |
| voltage magnitude | 0.0012 p.u. | absolute |
| voltage angle | 0.00168 rad | absolute |
| PMU current | 1/300 of the current's magnitude | relative |
| PMU current angle | 1/300 rad | absolute |

The PMU current class is the 1 % total vector error of IEEE C37.118.1 [C37118], taken as three
standard deviations (`PMU_CURRENT_CLASS`). It is relative to the phasor's magnitude, on the real
and the imaginary part alike.

A relative meter also has a floor, so a reading near zero still has noise: 0.001 MW or MVAr on a
power meter (`POWER_NOISE_FLOOR_MW`) and 0.00001 p.u. on a current (`CURRENT_FLOOR_PU`).

`MeterSettings.power_floor_pu` sets the power meters' floor for a data set, per unit of the system
base. None, the default, keeps 0.001 MW whatever the base. The SCADA noise of [WU26]'s case studies
is a flat 0.03 p.u. (`WU26_SCADA_NOISE_PU`), which is 3 MW on a 100 MVA base. The floor is part of
the noise every reading carries, so the ramp's stealth bound, the tamper count and the limit
tolerance follow it ([The attack families](attacks.md)). A file made with a floor of its own records
it in the attribute `power_floor_mw`, and `fg.load`, the clients' reader and the state estimator
weight by the stored value. A file without the attribute has the package's own floor.

## Bias and jitter

A meter's error is mostly a fixed calibration offset. The package splits each class in two.

| Part | Share of the class | Drawn |
|---|---|---|
| jitter | 0.25 (`JITTER_FRAC`) | again at every scan |
| bias | `sqrt(1 - 0.25^2)`, about 0.97 | once per meter, with the plan |

The squares of the two shares add to 1 (0.25² + 0.97² = 1), so bias and jitter together have
the standard deviation of the class. On a relative meter the bias is a gain error. On
an absolute meter it is an offset.

The biases are drawn with the plan: one draw for every bus and branch, metered or not
(`MeterPlan.from_placement`).

A reading is the exact value with its bias, plus the jitter of that scan (`NoiseModel.sample`).
`NoiseModel.sigmas(z)` gives the total of both, which an estimator weights by.

Treating the whole class as per-scan noise would make consecutive scans jump more than real
meters do, and would hide the slow changes a detector looks for.

## How the noise is keyed

The jitter of frame `t` comes from the frame's own random stream, keyed by the seed, a fixed
stream code (`JITTER_STREAM` in `constants/streams.py`) and the frame's position in the pool.

So the noise of a frame does not depend on which attacks are placed, nor on where the walk
stops and starts. Two timelines built from one seed and one pool with different families carry
the same benign scans. Only the attacks differ.

The biases are part of the plan, so they are fixed by the seed too.

## What is not in a file

A timeline file stores the meter masks, not the table. It does not store each meter's accuracy
class (the table `KIND_ACCURACY` gives it) or its bias (drawn from the seed).

## Where next

- [The attack families](attacks.md): what an attack adds to a scan.
- [Layers and labels](layers.md): how a scan is stored.
- [Formulas](../reference/FORMULAS.md): the noise equations and their sources.
