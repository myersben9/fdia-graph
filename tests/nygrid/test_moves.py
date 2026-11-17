"""The move rule on the New York grid, and the noise of a pool frame, on the synthetic grid of `conftest.py`
(no NYgrid data, no solver): a placed episode with no design moves inside its split and only where its
family's mask allows, is given up after the redraw limit, never moves when the survey already holds its
design; and one pool frame reads the same in a timeline and in a single-episode file."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from fdia_graph import schema
from fdia_graph.constants.families import FAMILY_CODE

from .conftest import LENGTHS, allowed_hours, bias_family

AM, AT = FAMILY_CODE["Am"], FAMILY_CODE["At"]


def _generate(grid, path, am=None, at=None, **settings):
    """The synthetic timeline with the families' options `am` and `at` (`bias_family`): (source, file facts)."""
    from fdia_graph.generation.nygrid.source import NewYorkSource
    from fdia_graph.models.config import NewYorkSettings

    pool, att, _, own = grid
    plan = att.meas.plan
    families = {"Am": bias_family("Am", plan, **(am or {})), "At": bias_family("At", plan, **(at or {}))}
    source = NewYorkSource(pool, att, families, own, NewYorkSettings(length=LENGTHS, **settings))
    source.generate(path)
    with h5py.File(path, "r") as h:
        episodes = h[schema.Group.EPISODES]
        facts = {
            "split": h[schema.SPLIT][:],
            "onset": episodes[schema.EPISODE_ONSET][:],
            "length": episodes[schema.EPISODE_LENGTH][:],
            "family": episodes[schema.EPISODE_FAMILY][:],
        }
    return source, facts


def _whole_and_apart(facts) -> bool:
    """Every episode lies in one split and no frame is under two episodes."""
    cover = np.zeros(len(facts["split"]), int)
    one_split = True
    for onset, length in zip(facts["onset"], facts["length"]):
        cover[onset : onset + length] += 1
        one_split &= len(set(facts["split"][onset : onset + length].tolist())) == 1
    return bool(one_split and cover.max() == 1)


def test_a_masked_family_moves_only_onto_its_mask_inside_its_split(grid, tmp_path):
    source, facts = _generate(grid, tmp_path / "masked.h5", am={"masked": True}, attacked_frac=0.3)
    summary = source.summary()
    am = facts["onset"][facts["family"] == AM]
    assert sum(row[0] for row in summary["moved"]) > 0  # some Am had no attack at its onset and moved
    assert len(am) > 5 and allowed_hours()[am].all()  # every Am, moved or not, starts where its mask allows
    assert not (facts["onset"] % 7 == 0).any()  # none is left where its family finds no attack
    assert _whole_and_apart(facts)


def test_an_episode_is_given_up_after_the_redraw_limit_and_the_summary_accounts_for_it(grid, tmp_path):
    """An attack only at one onset in eleven: most attempts miss, so some slots use up their moves."""
    from fdia_graph.generation.plan import REDRAWS

    hard = {"every": 11}
    with pytest.warns(RuntimeWarning, match=f"no feasible design after {REDRAWS} moves"):
        source, facts = _generate(grid, tmp_path / "hard.h5", am=hard, at=hard, attacked_frac=0.3)
    summary = source.summary()
    placed, built, given_up = (np.array(summary[k]) for k in ("placed", "built", "given_up"))
    moved = np.array(summary["moved"])
    assert given_up.sum() > 0 and (placed - built == given_up).all() and len(facts["onset"]) == built.sum()
    # every failed attempt moved its slot or, at the limit, gave it up
    misses = sum(sum(summary["missed"][f].values()) for f in LENGTHS) + summary["unsolved_attempts"]
    assert misses == moved.sum() + given_up.sum()
    with h5py.File(tmp_path / "hard.h5", "r") as h:  # the file records the same counts
        assert np.array_equal(h.attrs[schema.Attr.EPISODE_REDRAWS], moved)
        assert np.array_equal(h.attrs[schema.Attr.EPISODE_SHORTFALL], given_up)
    assert (facts["onset"] % 11 == 0).all() and _whole_and_apart(facts)


@pytest.mark.parametrize("split_by", ["week", "time"])
def test_a_surveyed_family_is_placed_only_on_survey_hits_and_never_moves(grid, tmp_path, split_by):
    settings = dict(survey=6, attacked_frac=0.3, split_by=split_by)
    source, facts = _generate(grid, tmp_path / "survey.h5", am={"masked": True}, **settings)
    summary = source.summary()
    hits = source.onset_masks()["Am"]  # after the survey: the onsets where it found an attack
    grid_onsets = np.flatnonzero(hits)
    assert hits.any() and (grid_onsets % 6 == 0).all() and allowed_hours()[grid_onsets].all()
    am = facts["onset"][facts["family"] == AM]
    assert len(am) >= 3 and hits[am].all()  # every Am sits on a hit
    assert [row[0] for row in summary["moved"]] == [0, 0, 0]  # its design was known: no Am ever moved
    assert sum(row[1] for row in summary["moved"]) > 0  # the family that starts anywhere still moves
    tried, found = np.array(list(summary["survey"].values())).sum(axis=0)
    assert found == hits.sum() < tried  # the survey tried more onsets than it found attacks at
    assert _whole_and_apart(facts)


def test_a_pool_frame_reads_the_same_in_a_timeline_and_in_a_single_episode_file(grid, tmp_path):
    """A frame's jitter is keyed by its pool frame, wherever the file starts."""
    from fdia_graph.generation.nygrid.source import SingleEpisodeSource

    source, _ = _generate(grid, tmp_path / "timeline.h5")
    files = {}
    for lo, hi in ((480, 540), (500, 560)):
        files[lo] = SingleEpisodeSource(source, 510, "At", lo, hi).generate(tmp_path / f"one_{lo}.h5")
    layers = (schema.NODE_BENIGN, schema.EDGE_BENIGN, schema.PMU_I_BENIGN, schema.NODE_CLEAN)
    with h5py.File(tmp_path / "timeline.h5", "r") as whole:
        for lo, path in files.items():
            with h5py.File(path, "r") as one:
                for name in layers:
                    assert np.array_equal(one[name][:], whole[name][lo : lo + 60]), name
    with h5py.File(files[480], "r") as a, h5py.File(files[500], "r") as b:  # two files, one pool hour
        assert np.array_equal(a[schema.NODE_BENIGN][20:], b[schema.NODE_BENIGN][:40])
        assert not np.array_equal(a[schema.NODE_BENIGN][:40], b[schema.NODE_BENIGN][:40])  # nor shared draws
