"""Explicit mapping from the upstream Kaggle schema to our local schema.

Written out by hand on purpose. An automatic name matcher looks tempting, but
the two schemas contain genuinely ambiguous pairs - locally `r_str_acc` is the
fighter's career striking accuracy while `r_total_str_acc` is that accuracy in
this bout, and they differ by a single "total" token. Guessing wrong there
would silently corrupt the training data with no error anywhere, so the map is
declared and tested rather than inferred.

Upstream layout (master.csv, one row per bout):
    r_fighter_name, r_total_sig_str_landed_head, r_total_ctrl_seconds, ...
Local layout:
    r_name,         r_head_landed,               r_ctrl,               ...
"""

# (upstream_position, local_position)
POSITIONS = [
    ('head', 'head'), ('body', 'body'), ('leg', 'leg'),
    ('distance', 'dist'), ('clinch', 'clinch'), ('ground', 'ground'),
]

# Per-corner, {upstream: local}. Formatted with the corner prefix.
PER_CORNER = {
    '{c}_fighter_name': '{c}_name',
    '{c}_fighter_id': '{c}_id',
    '{c}_fighter_nick_name': '{c}_nick_name',
    '{c}_total_kd': '{c}_kd',
    '{c}_total_sig_landed': '{c}_sig_str_landed',
    '{c}_total_sig_atmp': '{c}_sig_str_atmpted',
    '{c}_total_str_landed': '{c}_total_str_landed',
    '{c}_total_str_atmp': '{c}_total_str_atmpted',
    '{c}_total_td_success': '{c}_td_landed',
    '{c}_total_td_atmp': '{c}_td_atmpted',
    '{c}_total_sub_att': '{c}_sub_att',
    '{c}_total_ctrl_seconds': '{c}_ctrl',
    '{c}_reach_inches': '{c}_reach',
    '{c}_weight_lbs': '{c}_weight',
    '{c}_slpm': '{c}_splm',
}

# Bout-level columns. Names already shared are handled by IDENTICAL below.
BOUT = {
    'event_date': 'date',
    'event_location': 'location',
    'weight_class': 'division',
}

IDENTICAL = ['event_id', 'event_name', 'fight_id', 'winner_id', 'method', 'referee']

# Career-profile columns, joined from fighter.csv on fighter_id.
# {fighter.csv column: local suffix}
FIGHTER_PROFILE = {
    'height': 'height', 'dob': 'dob', 'stance': 'stance',
    'str_acc': 'str_acc', 'sapm': 'sapm', 'str_def': 'str_def',
    'td_avg': 'td_avg', 'td_acc': 'td_avg_acc', 'td_def': 'td_def',
    'sub_avg': 'sub_avg',
}

# Accuracy percentages: local column <- (landed, attempted)
def accuracy_columns(corner):
    cols = {
        f'{corner}_sig_str_acc': (f'{corner}_sig_str_landed', f'{corner}_sig_str_atmpted'),
        f'{corner}_total_str_acc': (f'{corner}_total_str_landed', f'{corner}_total_str_atmpted'),
        f'{corner}_td_acc': (f'{corner}_td_landed', f'{corner}_td_atmpted'),
    }
    for _, loc in POSITIONS:
        cols[f'{corner}_{loc}_acc'] = (f'{corner}_{loc}_landed', f'{corner}_{loc}_atmpted')
    return cols


# Share-of-significant-strikes percentages: local column <- (part, whole)
def share_columns(corner):
    return {
        f'{corner}_landed_{loc}_per': (f'{corner}_{loc}_landed', f'{corner}_sig_str_landed')
        for _, loc in POSITIONS
    }


def build_column_map():
    """{upstream master.csv column: local column} for straight renames."""
    out = {}
    for corner in ('r', 'b'):
        for up, loc in PER_CORNER.items():
            out[up.format(c=corner)] = loc.format(c=corner)
        for up_pos, loc_pos in POSITIONS:
            out[f'{corner}_total_sig_str_landed_{up_pos}'] = f'{corner}_{loc_pos}_landed'
            out[f'{corner}_total_sig_str_atmp_{up_pos}'] = f'{corner}_{loc_pos}_atmpted'
    out.update(BOUT)
    out.update({c: c for c in IDENTICAL})
    return out


COLUMN_MAP = build_column_map()

# Local columns we can compute rather than read.
DERIVED = sorted(
    set(accuracy_columns('r')) | set(accuracy_columns('b'))
    | set(share_columns('r')) | set(share_columns('b'))
)

# Local columns produced by ratings.py, not by the sync.
RATING_COLUMNS = ['r_mmr_pre', 'b_mmr_pre', 'r_mu_pre', 'b_mu_pre',
                  'r_sigma_pre', 'b_sigma_pre']

# Local columns with no upstream source. Win/loss/draw records are the
# fighter's overall professional record including bouts outside the UFC, which
# this dataset does not carry, so they are carried forward per fighter and
# incremented from observed results instead of being read.
CARRIED_FORWARD = ['r_wins', 'r_losses', 'r_draws', 'b_wins', 'b_losses', 'b_draws']

def profile_columns(corner):
    """{local column: fighter.csv column} for one corner's career profile."""
    return {f'{corner}_{loc}': up for up, loc in FIGHTER_PROFILE.items()}


# Local columns resolved from another column rather than renamed.
#   winner      <- winner_id matched against r_fighter_id / b_fighter_id
#   title_fight <- upstream marks title bouts in the weight_class text
RESOLVED = ['winner', 'title_fight']


# Needs a unit conversion rather than a rename.
CONVERTED = {
    'match_time_sec': 'finish_time',   # "4:31" -> 271
    'total_rounds': 'time_format',     # "3 Rnd (5-5-5)" -> 3
    'finish_round': 'rounds_fought',
}
