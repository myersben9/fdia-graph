"""The fixed numerical tolerances and finite-difference steps of the algorithms."""

from __future__ import annotations

# A magnitude at or below this is zero: a pivot, a singular value, an entry of an attack (the row
# reduction and the attack-subspace formulas of both toolkits).
ZERO_TOL = 1e-9
# A Jacobian entry or a state change this small relative to the largest is structurally zero (the
# row-reduction support of the overload attack).
RELATIVE_ZERO = 1e-9
JACOBIAN_V_STEP = 1e-6  # pu: the finite-difference step of |V| in the attack-area Jacobian
JACOBIAN_THETA_STEP = 1e-5  # degrees: the finite-difference step of the angle in the same Jacobian
BACKTRACK_HALVINGS = 8  # step fractions the Newton power flow tries per iteration: 1, 1/2, ... 1/128
# Slack under an operating limit, below what a meter resolves (0.0012 pu on |V|, 1e-3 MW on P and Q):
# a true state outside a limit is itself the bound (a regulated bus at its setpoint), and a false
# state that moves it by less than the meter floor violates nothing an operator could see. On P and
# Q this is the least slack: a generator limit is passed only by more than the accuracy of the bus's
# injection meter at its true reading (`formulas.attacks.limit_tolerance`), which is never below it.
LIMIT_TOL_V = 1e-3  # pu
LIMIT_TOL_PQ = 1e-3  # MW, MVAr
