# Plans

The design documents of the work in progress: what was decided, what was rejected, and why.
`CONTRIBUTING.md` says how to change the package today. Finished plans are removed once the code
and the changelog carry their decisions; the earlier ones (the readability, restructure, data-model,
field-group, validation, one-dataset and reference-study plans of 0.16 to 0.20) are in the git
history at commit `aa77d0b`.

| plan | what it decided |
|---|---|
| `WU_MSFDIA_PLAN.md` | **accepted**: generate the multi-snapshot families `At` and `Am` by solving the [WU26] optimization (fewest tampered devices, eq. 12) under its constraints and line-overload goal, and move all attack generation into the engine's attack mixin; decisions D1 to D17 |
| `WU_DEFENSE_PLAN.md` | **accepted, in progress**: [WU26]'s trusted-PMU defense as the paper defines it (the trust schedule on the fewest-tamper search, the row-reduction attack, Solution 1 and the DQN), with the reproduction harness |
| `RELAX_CERTIFIER_PLAN.md` | **shipped, optional** (`[certify]`): a mixed-integer second-order-cone relaxation of [WU26] eq. (12) that bounds the fewest-tamper search from below, so an attack whose count meets the bound is certified globally optimal over the attacker's area; on IEEE-14 it is valid but loose for `Am` |
