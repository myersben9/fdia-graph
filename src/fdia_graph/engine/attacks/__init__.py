"""Every attack the generator builds, one line per file.

area.py         `AttackArea`: given seed buses, the attacker's area and the boundary of any set of buses
false_state.py  `FalseState`: the false state of a true state, a support and a goal, inside the limits
frames.py       `AttackFrames`: the attack vector a = h(x_false) - h(x_true), added to the true scan
ramp_frames.py  `RampFrames`: one At frame, the false state of its `AttackDesign` written as a frame
overload.py     `Overload`: Am's ratings, eligible lines, flow schedule [WU26 eqs. 24-25] and one frame
window.py       `AttackWindow`: what one candidate support costs over a window
blas.py         one BLAS thread for the length of a search
minimize.py     `Minimizer`: the search for the cheapest support [WU26 eq. 12], behind `min_tamper`
rref.py         `RrefSupport`: the other way to choose a support, [WU26]'s row reduction
episodes.py     `EpisodeDesigns`: the design of one episode, `ramp_design` for At, `am_overload_design` for Am
family.py       what the timeline asks of any family, and the base of a family of your own
search.py       At and Am as families: `EpisodeDesigns`, `RampFrames` and `Overload` called frame by frame

This is the order to read the files in. `minimize.py` and `rref.py` call each other.
"""
