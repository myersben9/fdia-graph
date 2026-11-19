"""Every attack the generator builds. Seven files each hold one part of the generator; four hold what
serves or uses the parts.

area.py         `AttackArea`: where a stealthy attack acts, the interior it moves and the boundary it holds true
false_state.py  `FalseState`: the local false state of a load goal or a flow goal, inside its operating limits
frames.py       `AttackFrames`: the attack vector a = h(x_false) - h(x_true), written onto the true scan
ramp_frames.py  `RampFrames`: the frames of the ramp At
minimize.py     `Minimizer`: the fewest-tamper support of an attack window [WU26 eq. 12], behind `min_tamper`
overload.py     `Overload`: Am, the overload attack of [WU26]: ratings, target branches, flow goal, frames
episodes.py     `EpisodeDesigns`: what an episode attacks, `ramp_design` for At and `am_overload_design` for Am

window.py       `AttackWindow`: one attack window, and what a candidate support costs over it
rref.py         `RrefSupport`: [WU26]'s row reduction as the minimizer's other choice of a support
search.py       the two families as the design stage asks them, and their episodes as the walk emits them
family.py       the interface every family shares, and the base of a family of your own
"""
