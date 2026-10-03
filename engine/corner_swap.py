"""The same fight with the corners exchanged, as a model row.

The red corner is usually the favoured or ranked fighter (the red corner has
won 58% of post-2001 UFC bouts), so a model fitted on rows as listed can
learn corner identity as well as fighters. Testing whether that costs
anything needs the swapped row of every fight - what the pipeline would
have built had the card listed the fighters the other way round - and
production, if symmetrisation ships, needs the same row for a live fight.
Both come from here.

HOW A FEATURE MIRRORS (measured over the whole archive, experiments/
symmetry.py writes the table): with the corner-swapped archive run through
feature_frame.build and the interaction centres fitted on that frame,
every one of the 151 features is a clean mirror - 71 unchanged (levels,
known flags, the fight's own flags), 66 negated (every red-minus-blue
difference, the matchup advantages, archetype_matchup), 2 flipped as 1 - x
(bayesian_prob, base_prob) and 12 exchanged with a twin (the four
trajectory columns and the eight r_*_vs_b_* interactions). The target
flips by itself, because `winner` names the fighter and target_win is
`winner == r_name`.

Two things are not that simple, and both matter for a LIVE swapped row,
which is built on the archive as it is with the training centres frozen:

  * The eight Interaction features are centred on the training frame's
    corner means, and those differ by corner (red td_avg 1.79 against blue
    1.68; red has_been_kod 0.33 against blue 0.41 - the red corner is the
    better fighter). With the centres frozen, r_td_vs_b_tdd of the swapped
    row is (b_td_avg - red centre)(r_td_def - blue centre), which is not
    b_td_vs_r_tdd of the row as listed. The swapped row is built with the
    frozen centres anyway, because that is the row production can build.
  * Three features count a fighter's EARLIER APPEARANCES IN THAT CORNER
    (feature_frame: r_fight_count = groupby('r_name').cumcount()):
    data_sparsity_diff, career_damage_diff and career_damage_level. They
    mirror under a whole-archive swap only because the history is swapped
    with them; for one fight swapped on the real history they read the
    other corner's count. They are, in other words, corner-identity
    features already: a fighter's red-corner appearances are a count of
    how often the matchmaker favoured them. swapped_matrix recomputes them
    the way the live swapped row reads them, so the two agree exactly.

What tests/test_symmetry.py proves about the live swapped row: on the last
archived event it equals swapped_matrix's row on every feature. On earlier
dates both orientations of the live row differ from their training rows on
the same rating-derived features (mu_diff, mmr_diff, mx_mass_adv and their
relatives) by the same amounts - the archive's shipped rating series are
not reproduced bit for bit by a replay, which predates and is independent
of the swap - while the swapped live row is the exact mirror of the listed
live row on every feature outside CORNER_HISTORY and the eight frozen-
centre interactions, and equals swapped_matrix's row on CORNER_HISTORY.

    mirror_archive(raw)            -> raw with r_* and b_* exchanged
    swapped_matrix(raw, built)     -> X of every row of built, corners exchanged
"""

import numpy as np
import pandas as pd

import feature_frame

# The features that count a corner's earlier appearances, and so differ
# between a whole-archive mirror and a single fight swapped on the history
# as it is.
CORNER_HISTORY = ("data_sparsity_diff", "career_damage_diff", "career_damage_level")
FIGHT_COUNT_CAP = 30         # feature_frame: r_fight_count.clip(upper=30)
DAMAGE_MINUTES = 12.0        # feature_frame: sapm * 12.0 * fights
DAMAGE_SAPM_DEFAULT = 3.0    # feature_frame: r_sapm.fillna(3)


def mirror_archive(raw):
    """`raw` with every r_* column exchanged for its b_* twin, in the same
    column order; columns without a corner prefix untouched. Refuses a
    column with no twin rather than leave one corner's value in place."""
    rename = {}
    for col in raw.columns:
        if col.startswith("r_"):
            twin = "b_" + col[2:]
        elif col.startswith("b_"):
            twin = "r_" + col[2:]
        else:
            continue
        if twin not in raw.columns:
            raise KeyError(f"{col!r} has no corner twin {twin!r}")
        rename[col] = twin
    return raw.rename(columns=rename)[list(raw.columns)]


def corner_appearances(red_names, blue_names):
    """For each row, (earlier red-corner rows of this row's BLUE fighter,
    earlier blue-corner rows of this row's RED fighter): what r_fight_count
    and b_fight_count read for the fight swapped on the history as it is.
    Counted by name in row order, exactly as feature_frame's
    groupby(name).cumcount() counts them."""
    red_seen, blue_seen = {}, {}
    red_of_blue = np.empty(len(red_names), dtype=int)
    blue_of_red = np.empty(len(red_names), dtype=int)
    for i, (r, b) in enumerate(zip(red_names, blue_names)):
        red_of_blue[i] = red_seen.get(b, 0)
        blue_of_red[i] = blue_seen.get(r, 0)
        red_seen[r] = red_seen.get(r, 0) + 1
        blue_seen[b] = blue_seen.get(b, 0) + 1
    return red_of_blue, blue_of_red


def corner_history_features(ufc):
    """The three CORNER_HISTORY features of every row of `ufc` (a built
    frame, as listed) for the SWAPPED fight on the unchanged history, with
    feature_frame's own arithmetic: log1p of the capped count, and log1p of
    sapm * 12 * max(count, 1) for the damage proxy."""
    red_cnt, blue_cnt = corner_appearances(ufc["r_name"].to_numpy(),
                                           ufc["b_name"].to_numpy())
    # the swapped row's red fighter is the listed blue fighter, so its sapm
    # is b_sapm, and the other way round
    r_sapm = pd.to_numeric(ufc["b_sapm"], errors="coerce").fillna(DAMAGE_SAPM_DEFAULT).to_numpy(float)
    b_sapm = pd.to_numeric(ufc["r_sapm"], errors="coerce").fillna(DAMAGE_SAPM_DEFAULT).to_numpy(float)
    r_rel = np.log1p(np.minimum(red_cnt, FIGHT_COUNT_CAP))
    b_rel = np.log1p(np.minimum(blue_cnt, FIGHT_COUNT_CAP))
    r_dl = np.log1p(r_sapm * DAMAGE_MINUTES * np.maximum(red_cnt, 1))
    b_dl = np.log1p(b_sapm * DAMAGE_MINUTES * np.maximum(blue_cnt, 1))
    return pd.DataFrame({"data_sparsity_diff": r_rel - b_rel,
                         "career_damage_diff": r_dl - b_dl,
                         "career_damage_level": (r_dl + b_dl) / 2.0},
                        index=ufc.index)


def swapped_matrix(raw, built, verbose=False):
    """X (every feature column of `built`, same index and row order) for
    every row of built["ufc"] with the corners exchanged and the history
    as it is: the mirrored archive through feature_frame.build with the
    training centres frozen, and the CORNER_HISTORY features recomputed.
    This is the row live_rows builds for the swapped fight - equal to it on
    every feature on the last archived event, and on earlier dates up to
    the rating-replay drift both orientations share (module docstring)."""
    mirrored = feature_frame.build(mirror_archive(raw),
                                   centres=built["INTERACTION_CENTRES"],
                                   verbose=verbose)
    ufc, mufc = built["ufc"], mirrored["ufc"]
    if (len(mufc) != len(ufc)
            or not np.array_equal(mufc["fight_id"].astype(str).to_numpy(),
                                  ufc["fight_id"].astype(str).to_numpy())):
        raise RuntimeError("the mirrored archive did not come back in the "
                           "training row order - no swapped row can be trusted")
    if not np.array_equal(mufc["r_name"].to_numpy(), ufc["b_name"].to_numpy()):
        raise RuntimeError("the mirrored frame's red corner is not the blue corner")
    X = mirrored["X"][built["feature_cols"]].copy()
    X.index = built["X"].index
    patch = corner_history_features(ufc)
    for col in CORNER_HISTORY:
        X[col] = patch[col].to_numpy()
    return X.replace([np.inf, -np.inf], np.nan).fillna(0)


def symmetrise(p_as_listed, p_swapped):
    """(p(x) + 1 - p(x_swapped)) / 2, elementwise: the one number that is
    the same fight either way round."""
    return (np.asarray(p_as_listed, float) + 1.0 - np.asarray(p_swapped, float)) / 2.0
