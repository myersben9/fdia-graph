"""The meters: what each client measures, and how wrong each meter is.

`MeterPlan.from_coverage` places SCADA meters (injections, branch flows, voltage magnitudes) and PMUs on a
case, and a source that knows who owns each bus gives each meter to that client: the `MeterPlan`, a table
beside the flat measurement vector `z`, so a client's readings are `z[:, plan.owner == c]`. `measure.Measurement` evaluates h(x) and its Jacobian for a state, and
`noise.NoiseModel` gives each meter its accuracy class split into a constant bias and a per-scan jitter.
"""
