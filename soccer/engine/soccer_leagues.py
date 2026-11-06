"""Which leagues the app carries, and what it shows for each - as data.

experiments/leagues.py measured every league football-data.co.uk carries
(soccer_data.ALL_LEAGUES), walk-forward, and decided by rules declared
before any number was seen (its docstring has the protocol, every table and
the caveats). This module is what it decided, and the rules it decided by:
the experiment imports the rules below, and a test re-applies them to the
numbers stored here, so a decision can never drift from its measurement.

THE RULES (rule_* below):
  (a) in the app   a current season in the data (a result in the 45 days
                   before the check, or a fixture posted) - rule_current;
                   the pick is the market recalibrated if that was not worse
                   in log loss than the raw no-vig market on the league's
                   development period, else the raw no-vig market - rule_pick
  (b) model alone  a match with no price yet gets the model's own pick only
                   in a league where the model beat the league's base rate
                   (its home / draw / away frequencies over earlier seasons)
                   in log loss in BOTH periods; elsewhere "no pick until
                   priced" - rule_model_alone
  (c) numbers      both teams to score, over 1.5 / 2.5 / 3.5 and the exact
                   score each ship for a league only where the source the
                   card would read them from beat the league's base rate in
                   log loss in BOTH periods - the market-implied matrix
                   ("market", where the 1X2 and over/under are priced) and
                   the model's own ("model") judged apart - rule_numbers
  (d) bands        edges 0.40 / 0.50 / 0.60 as before; hit rates measured
                   per league group on the confirmation period - BANDS

HOW THE PIPELINE USES IT. smod.predict(frame, leagues=MODEL_LEAGUES) - the
model over every league measured (England's pool ENGLAND_POOL), so each
group's regression is the one measured (RUS's matches train the extra
leagues' even while RUS is not shown); smod.shipped_pick(frame, preds) -
pick_leagues(frame, preds, in_app(), PICK_WEIGHTS, raw=raw_leagues(),
model_alone=model_alone()); smod.goals(frame, preds, in_app()) and
ssc.walk(frame, preds, pick, leagues=in_app()); ssc.card(...,
withhold=withheld(code, source)); band_rate(code, band, basis);
caveats(code) on every card and on the Method tab. Until the pipeline
switches (soccer_predict.py still runs the five top leagues as before),
none of this changes what the app shows.
"""

import soccer_data as sd

CHECKED_ON = "2026-11-02"           # the day rule (a)'s current-season check was run
RECENT_DAYS = 45
NUMBERS = ("btts", "o15", "o25", "o35", "exact")
SOURCES = ("market", "model")
GROUPS = {
    "top": list(sd.TOP),
    "europe": [c for c in sd.MAIN if c not in sd.TOP],
    "extra": list(sd.EXTRA),
}
GROUP_NAMES = {"top": "top five", "europe": "other European divisions", "extra": "extra leagues"}
MODEL_LEAGUES = list(sd.ALL_LEAGUES)      # the model runs over every league it was measured on


def group_of(code):
    """'top', 'europe' or 'extra'."""
    return next(g for g, codes in GROUPS.items() if code in codes)


# ----------------------------------------------------------- the rules

def rule_current(days_since_result, fixtures_posted, days=RECENT_DAYS):
    """(a): a current season in the data - a result within `days` of the
    check (days_since_result: None where the league has none), or any
    fixture posted for it."""
    return bool((days_since_result is not None and days_since_result <= days) or fixtures_posted > 0)


def season_check(matches, fixtures, code, today=CHECKED_ON, days=RECENT_DAYS):
    """(a)'s facts for one league: its newest result and how many days
    before `today` it was played, and the fixtures posted for it (on file -
    football-data's fixture files list matches not yet played - dated no
    earlier than `days` before `today`); "ok" is rule_current on them."""
    import pandas as pd
    res = matches[(matches["league"] == code) & matches["result"].notna()]
    last = res["date"].max() if len(res) else None
    since = (pd.Timestamp(today) - pd.Timestamp(last)).days if last else None
    since_day = (pd.Timestamp(today) - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    posted = int(((fixtures["league"] == code) & (fixtures["date"] >= since_day)).sum()) if len(fixtures) else 0
    return {"ok": rule_current(since, posted, days), "last_result": last, "days_since": since,
            "fixtures_posted": posted, "season": int(res["season"].max()) if len(res) else None}


def rule_pick(dev_recalibrated, dev_raw):
    """(a): 'recalibrated' unless the recalibrated market's development log
    loss is worse (higher) than the raw no-vig market's on the same matches;
    then 'raw'. Nothing measured: 'raw'."""
    if dev_recalibrated is None or dev_raw is None:
        return "raw"
    return "recalibrated" if dev_recalibrated <= dev_raw else "raw"


def rule_model_alone(dev, confirm):
    """(b): True where the model's log loss beat the base rate's ({"model",
    "base"} per period, same matches) in BOTH periods."""
    return all(p.get("model") is not None and p.get("base") is not None and p["model"] < p["base"]
               for p in (dev, confirm))


def rule_numbers(dev, confirm):
    """(c): the NUMBERS a source ships for a league - each whose log loss beat
    the base rate's in BOTH periods. dev / confirm: {number: {"log_loss",
    "base"}} (a period with no matches: {} - nothing ships)."""
    def beats(p, k):
        x = p.get(k) or {}
        return x.get("log_loss") is not None and x.get("base") is not None and x["log_loss"] < x["base"]
    return [k for k in NUMBERS if beats(dev, k) and beats(confirm, k)]


# ----------------------------------------------------------- the decisions
# Filled from experiments/leagues.py's run (its main() prints this block).

# Generated by experiments/leagues.py (main) on the run of 2026-11-02; do not edit
# by hand - a test re-applies the rules to the numbers stored here.
ENGLAND_POOL = 'top-two'
PICK_WEIGHTS = 'per_price'
SHIPPED = {
    'E0': {
        'name': 'Premier League',
        'country': 'England',
        'group': 'top',
        'in_app': True,
        'pick': 'recalibrated',
        'model_alone': True,
        'numbers': {'market': ['btts', 'o15', 'o25', 'o35', 'exact'], 'model': ['btts', 'o15', 'o25', 'o35', 'exact']},
        'weights': {'season': 2026, 'w': [1.131, 0.0426, 0.0107]},
        'caveats': [
            'On 2016-2025 the recalibrated pick scored a little worse than the raw market (0.9522 against 0.9520).',
        ],
        'measured': {
            'price': 'pre-match',
            'season_check': {'last_result': '2026-10-25', 'days_since': 8, 'fixtures_posted': 10},
            'dev': {'seasons': [2006, 2015], 'matches': 3800, 'pick': 0.96757, 'pick_recalibrated': 0.96757, 'market': 0.96793, 'model': 0.97038, 'base': 1.06562, 'acc_pick': 0.5361, 'acc_market': 0.5361, 'acc_model': 0.5332},
            'confirm': {'seasons': [2016, 2025], 'matches': 3800, 'pick': 0.95222, 'pick_recalibrated': 0.95222, 'market': 0.95202, 'model': 0.96092, 'base': 1.06635, 'acc_pick': 0.5558, 'acc_market': 0.5563, 'acc_model': 0.5439},
            'scores': {
                'dev': {
                    'market': {'btts': {'log_loss': 0.68634, 'base': 0.69455}, 'o15': {'log_loss': 0.56539, 'base': 0.57209}, 'o25': {'log_loss': 0.68535, 'base': 0.69421}, 'o35': {'log_loss': 0.59179, 'base': 0.59996}, 'exact': {'log_loss': 2.8283, 'base': 2.96345}},
                    'model': {'btts': {'log_loss': 0.68908, 'base': 0.69455}, 'o15': {'log_loss': 0.56881, 'base': 0.57209}, 'o25': {'log_loss': 0.69046, 'base': 0.69421}, 'o35': {'log_loss': 0.59533, 'base': 0.59996}, 'exact': {'log_loss': 2.83744, 'base': 2.96345}},
                },
                'confirm': {
                    'market': {'btts': {'log_loss': 0.68663, 'base': 0.69261}, 'o15': {'log_loss': 0.51279, 'base': 0.52589}, 'o25': {'log_loss': 0.67492, 'base': 0.69127}, 'o35': {'log_loss': 0.61167, 'base': 0.62809}, 'exact': {'log_loss': 2.88804, 'base': 3.0387}},
                    'model': {'btts': {'log_loss': 0.68652, 'base': 0.69261}, 'o15': {'log_loss': 0.51503, 'base': 0.52589}, 'o25': {'log_loss': 0.68063, 'base': 0.69127}, 'o35': {'log_loss': 0.61576, 'base': 0.62809}, 'exact': {'log_loss': 2.89921, 'base': 3.0387}},
                },
            },
        },
    },
    'SP1': {
        'name': 'La Liga',
        'country': 'Spain',
        'group': 'top',
        'in_app': True,
        'pick': 'recalibrated',
        'model_alone': True,
        'numbers': {'market': ['btts', 'o15', 'o25', 'o35', 'exact'], 'model': ['btts', 'o15', 'o25', 'o35', 'exact']},
        'weights': {'season': 2026, 'w': [1.1308, 0.0426, 0.0105]},
        'caveats': [
        ],
        'measured': {
            'price': 'pre-match',
            'season_check': {'last_result': '2026-10-26', 'days_since': 7, 'fixtures_posted': 10},
            'dev': {'seasons': [2006, 2015], 'matches': 3800, 'pick': 0.95141, 'pick_recalibrated': 0.95141, 'market': 0.95256, 'model': 0.95871, 'base': 1.04942, 'acc_pick': 0.5511, 'acc_market': 0.5503, 'acc_model': 0.5416},
            'confirm': {'seasons': [2016, 2025], 'matches': 3800, 'pick': 0.96591, 'pick_recalibrated': 0.96591, 'market': 0.96734, 'model': 0.97618, 'base': 1.0676, 'acc_pick': 0.5371, 'acc_market': 0.5382, 'acc_model': 0.5292},
            'scores': {
                'dev': {
                    'market': {'btts': {'log_loss': 0.68881, 'base': 0.69334}, 'o15': {'log_loss': 0.55456, 'base': 0.56784}, 'o25': {'log_loss': 0.6732, 'base': 0.69478}, 'o35': {'log_loss': 0.57807, 'base': 0.60531}, 'exact': {'log_loss': 2.83633, 'base': 3.00033}},
                    'model': {'btts': {'log_loss': 0.68943, 'base': 0.69335}, 'o15': {'log_loss': 0.55546, 'base': 0.56805}, 'o25': {'log_loss': 0.67305, 'base': 0.69477}, 'o35': {'log_loss': 0.57825, 'base': 0.60522}, 'exact': {'log_loss': 2.84423, 'base': 3.00023}},
                },
                'confirm': {
                    'market': {'btts': {'log_loss': 0.68506, 'base': 0.69299}, 'o15': {'log_loss': 0.56982, 'base': 0.58288}, 'o25': {'log_loss': 0.66881, 'base': 0.69313}, 'o35': {'log_loss': 0.55377, 'base': 0.57329}, 'exact': {'log_loss': 2.77674, 'base': 2.91786}},
                    'model': {'btts': {'log_loss': 0.68636, 'base': 0.69299}, 'o15': {'log_loss': 0.57173, 'base': 0.58288}, 'o25': {'log_loss': 0.67189, 'base': 0.69313}, 'o35': {'log_loss': 0.55628, 'base': 0.57329}, 'exact': {'log_loss': 2.78884, 'base': 2.91786}},
                },
            },
        },
    },
    'I1': {
        'name': 'Serie A',
        'country': 'Italy',
        'group': 'top',
        'in_app': True,
        'pick': 'recalibrated',
        'model_alone': True,
        'numbers': {'market': ['o15', 'o25', 'o35', 'exact'], 'model': ['btts', 'o15', 'o25', 'o35', 'exact']},
        'weights': {'season': 2026, 'w': [1.131, 0.0426, 0.0106]},
        'caveats': [
            "Not shown from the market's prices: both teams to score - it did not beat the league's base rate in both 2006-2015 and 2016-2025.",
        ],
        'measured': {
            'price': 'pre-match',
            'season_check': {'last_result': '2026-10-29', 'days_since': 4, 'fixtures_posted': 10},
            'dev': {'seasons': [2006, 2015], 'matches': 3799, 'pick': 0.97498, 'pick_recalibrated': 0.97498, 'market': 0.97703, 'model': 0.98707, 'base': 1.06265, 'acc_pick': 0.532, 'acc_market': 0.532, 'acc_model': 0.5265},
            'confirm': {'seasons': [2016, 2025], 'matches': 3799, 'pick': 0.94657, 'pick_recalibrated': 0.94657, 'market': 0.94908, 'model': 0.95863, 'base': 1.08385, 'acc_pick': 0.5575, 'acc_market': 0.5562, 'acc_model': 0.5554},
            'scores': {
                'dev': {
                    'market': {'btts': {'log_loss': 0.69436, 'base': 0.69358}, 'o15': {'log_loss': 0.57255, 'base': 0.57889}, 'o25': {'log_loss': 0.68467, 'base': 0.69208}, 'o35': {'log_loss': 0.57487, 'base': 0.58268}, 'exact': {'log_loss': 2.80835, 'base': 2.91242}},
                    'model': {'btts': {'log_loss': 0.69131, 'base': 0.69358}, 'o15': {'log_loss': 0.57292, 'base': 0.57889}, 'o25': {'log_loss': 0.68578, 'base': 0.69208}, 'o35': {'log_loss': 0.57705, 'base': 0.58268}, 'exact': {'log_loss': 2.81396, 'base': 2.91242}},
                },
                'confirm': {
                    'market': {'btts': {'log_loss': 0.68561, 'base': 0.69102}, 'o15': {'log_loss': 0.53758, 'base': 0.54824}, 'o25': {'log_loss': 0.67918, 'base': 0.69511}, 'o35': {'log_loss': 0.59001, 'base': 0.60346}, 'exact': {'log_loss': 2.81846, 'base': 2.98689}},
                    'model': {'btts': {'log_loss': 0.68778, 'base': 0.69103}, 'o15': {'log_loss': 0.54038, 'base': 0.54846}, 'o25': {'log_loss': 0.68529, 'base': 0.69511}, 'o35': {'log_loss': 0.59394, 'base': 0.60339}, 'exact': {'log_loss': 2.83157, 'base': 2.98677}},
                },
            },
        },
    },
    'D1': {
        'name': 'Bundesliga',
        'country': 'Germany',
        'group': 'top',
        'in_app': True,
        'pick': 'raw',
        'model_alone': True,
        'numbers': {'market': ['btts', 'o15', 'o25', 'o35', 'exact'], 'model': ['btts', 'o15', 'o25', 'o35', 'exact']},
        'weights': None,
        'caveats': [
            'The recalibrated market lost to the raw market on 2006-2015 (1.0024 against 1.0006), so its pick is the raw no-vig market.',
        ],
        'measured': {
            'price': 'pre-match',
            'season_check': {'last_result': '2026-10-25', 'days_since': 8, 'fixtures_posted': 9},
            'dev': {'seasons': [2006, 2015], 'matches': 3060, 'pick': 1.00062, 'pick_recalibrated': 1.00239, 'market': 1.00062, 'model': 1.00578, 'base': 1.06769, 'acc_pick': 0.5078, 'acc_market': 0.5078, 'acc_model': 0.5085},
            'confirm': {'seasons': [2016, 2025], 'matches': 3060, 'pick': 0.97875, 'pick_recalibrated': 0.9798, 'market': 0.97875, 'model': 0.99151, 'base': 1.07017, 'acc_pick': 0.5248, 'acc_market': 0.5248, 'acc_model': 0.5147},
            'scores': {
                'dev': {
                    'market': {'btts': {'log_loss': 0.68292, 'base': 0.68855}, 'o15': {'log_loss': 0.51112, 'base': 0.5184}, 'o25': {'log_loss': 0.67781, 'base': 0.68835}, 'o35': {'log_loss': 0.6253, 'base': 0.63624}, 'exact': {'log_loss': 2.93918, 'base': 3.05426}},
                    'model': {'btts': {'log_loss': 0.68556, 'base': 0.68855}, 'o15': {'log_loss': 0.51237, 'base': 0.5184}, 'o25': {'log_loss': 0.67962, 'base': 0.68835}, 'o35': {'log_loss': 0.62631, 'base': 0.63624}, 'exact': {'log_loss': 2.95058, 'base': 3.05426}},
                },
                'confirm': {
                    'market': {'btts': {'log_loss': 0.66862, 'base': 0.67912}, 'o15': {'log_loss': 0.45182, 'base': 0.46342}, 'o25': {'log_loss': 0.65524, 'base': 0.6768}, 'o35': {'log_loss': 0.64025, 'base': 0.66894}, 'exact': {'log_loss': 2.97732, 'base': 3.11826}},
                    'model': {'btts': {'log_loss': 0.67111, 'base': 0.67912}, 'o15': {'log_loss': 0.45416, 'base': 0.46342}, 'o25': {'log_loss': 0.65883, 'base': 0.6768}, 'o35': {'log_loss': 0.64345, 'base': 0.66894}, 'exact': {'log_loss': 2.99887, 'base': 3.11826}},
                },
            },
        },
    },
    'F1': {
        'name': 'Ligue 1',
        'country': 'France',
        'group': 'top',
        'in_app': True,
        'pick': 'recalibrated',
        'model_alone': True,
        'numbers': {'market': ['btts', 'o15', 'o25', 'o35', 'exact'], 'model': ['btts', 'o15', 'o25', 'o35', 'exact']},
        'weights': {'season': 2026, 'w': [1.131, 0.0426, 0.0107]},
        'caveats': [
        ],
        'measured': {
            'price': 'pre-match',
            'season_check': {'last_result': '2026-10-26', 'days_since': 7, 'fixtures_posted': 9},
            'dev': {'seasons': [2006, 2015], 'matches': 3800, 'pick': 1.01337, 'pick_recalibrated': 1.01337, 'market': 1.01371, 'model': 1.0207, 'base': 1.07007, 'acc_pick': 0.4992, 'acc_market': 0.5016, 'acc_model': 0.4908},
            'confirm': {'seasons': [2016, 2025], 'matches': 3477, 'pick': 0.9822, 'pick_recalibrated': 0.9822, 'market': 0.98276, 'model': 0.99429, 'base': 1.07558, 'acc_pick': 0.5246, 'acc_market': 0.5243, 'acc_model': 0.5191},
            'scores': {
                'dev': {
                    'market': {'btts': {'log_loss': 0.69324, 'base': 0.69467}, 'o15': {'log_loss': 0.60236, 'base': 0.60797}, 'o25': {'log_loss': 0.67735, 'base': 0.6865}, 'o35': {'log_loss': 0.53045, 'base': 0.53879}, 'exact': {'log_loss': 2.73945, 'base': 2.82121}},
                    'model': {'btts': {'log_loss': 0.69274, 'base': 0.69467}, 'o15': {'log_loss': 0.60337, 'base': 0.60797}, 'o25': {'log_loss': 0.67867, 'base': 0.6865}, 'o35': {'log_loss': 0.53104, 'base': 0.53879}, 'exact': {'log_loss': 2.74611, 'base': 2.82121}},
                },
                'confirm': {
                    'market': {'btts': {'log_loss': 0.68758, 'base': 0.69474}, 'o15': {'log_loss': 0.54267, 'base': 0.56107}, 'o25': {'log_loss': 0.67229, 'base': 0.70288}, 'o35': {'log_loss': 0.58776, 'base': 0.61433}, 'exact': {'log_loss': 2.85127, 'base': 3.00213}},
                    'model': {'btts': {'log_loss': 0.68874, 'base': 0.69474}, 'o15': {'log_loss': 0.54549, 'base': 0.56107}, 'o25': {'log_loss': 0.67638, 'base': 0.70288}, 'o35': {'log_loss': 0.59191, 'base': 0.61433}, 'exact': {'log_loss': 2.86775, 'base': 3.00213}},
                },
            },
        },
    },
    'E1': {
        'name': 'Championship',
        'country': 'England',
        'group': 'europe',
        'in_app': True,
        'pick': 'raw',
        'model_alone': True,
        'numbers': {'market': ['btts', 'o15', 'o25', 'o35', 'exact'], 'model': ['btts', 'o15', 'o25', 'o35', 'exact']},
        'weights': None,
        'caveats': [
            'The recalibrated market lost to the raw market on 2006-2015 (1.0489 against 1.0485), so its pick is the raw no-vig market.',
        ],
        'measured': {
            'price': 'pre-match',
            'season_check': {'last_result': '2026-10-24', 'days_since': 9, 'fixtures_posted': 12},
            'dev': {'seasons': [2006, 2015], 'matches': 5520, 'pick': 1.04851, 'pick_recalibrated': 1.04894, 'market': 1.04851, 'model': 1.05345, 'base': 1.07792, 'acc_pick': 0.4553, 'acc_market': 0.4553, 'acc_model': 0.4487},
            'confirm': {'seasons': [2016, 2025], 'matches': 5520, 'pick': 1.03365, 'pick_recalibrated': 1.03358, 'market': 1.03365, 'model': 1.04387, 'base': 1.07595, 'acc_pick': 0.4754, 'acc_market': 0.4754, 'acc_model': 0.4678},
            'scores': {
                'dev': {
                    'market': {'btts': {'log_loss': 0.69004, 'base': 0.69164}, 'o15': {'log_loss': 0.57667, 'base': 0.57941}, 'o25': {'log_loss': 0.68846, 'base': 0.69314}, 'o35': {'log_loss': 0.56956, 'base': 0.57482}, 'exact': {'log_loss': 2.83892, 'base': 2.88585}},
                    'model': {'btts': {'log_loss': 0.69088, 'base': 0.69166}, 'o15': {'log_loss': 0.5771, 'base': 0.57955}, 'o25': {'log_loss': 0.6895, 'base': 0.69312}, 'o35': {'log_loss': 0.57078, 'base': 0.57476}, 'exact': {'log_loss': 2.84509, 'base': 2.88583}},
                },
                'confirm': {
                    'market': {'btts': {'log_loss': 0.68943, 'base': 0.69338}, 'o15': {'log_loss': 0.58208, 'base': 0.58549}, 'o25': {'log_loss': 0.68679, 'base': 0.69206}, 'o35': {'log_loss': 0.55172, 'base': 0.55779}, 'exact': {'log_loss': 2.80779, 'base': 2.8681}},
                    'model': {'btts': {'log_loss': 0.69016, 'base': 0.69338}, 'o15': {'log_loss': 0.58336, 'base': 0.58549}, 'o25': {'log_loss': 0.6887, 'base': 0.69206}, 'o35': {'log_loss': 0.55376, 'base': 0.55779}, 'exact': {'log_loss': 2.82166, 'base': 2.8681}},
                },
            },
        },
    },
    'SP2': {
        'name': 'Segunda División',
        'country': 'Spain',
        'group': 'europe',
        'in_app': True,
        'pick': 'recalibrated',
        'model_alone': True,
        'numbers': {'market': ['o15', 'o25', 'o35', 'exact'], 'model': ['o15', 'o25', 'o35', 'exact']},
        'weights': {'season': 2026, 'w': [1.1308, 0.0425, 0.0105]},
        'caveats': [
            'No shots in 2006-07 to 2016-17: the model rated it on goals alone in those seasons.',
            "Not shown from the market's prices: both teams to score - it did not beat the league's base rate in both 2006-2015 and 2016-2025.",
            "Not shown from the model alone: both teams to score - it did not beat the league's base rate in both 2006-2015 and 2016-2025.",
        ],
        'measured': {
            'price': 'pre-match',
            'season_check': {'last_result': '2026-10-26', 'days_since': 7, 'fixtures_posted': 11},
            'dev': {'seasons': [2006, 2015], 'matches': 4610, 'pick': 1.03715, 'pick_recalibrated': 1.03715, 'market': 1.03801, 'model': 1.04879, 'base': 1.06464, 'acc_pick': 0.4694, 'acc_market': 0.4701, 'acc_model': 0.4638},
            'confirm': {'seasons': [2016, 2025], 'matches': 4599, 'pick': 1.03699, 'pick_recalibrated': 1.03699, 'market': 1.03816, 'model': 1.04782, 'base': 1.06785, 'acc_pick': 0.4705, 'acc_market': 0.4686, 'acc_model': 0.4594},
            'scores': {
                'dev': {
                    'market': {'btts': {'log_loss': 0.69394, 'base': 0.6935}, 'o15': {'log_loss': 0.61039, 'base': 0.61337}, 'o25': {'log_loss': 0.68103, 'base': 0.68726}, 'o35': {'log_loss': 0.54924, 'base': 0.555}, 'exact': {'log_loss': 2.78853, 'base': 2.8337}},
                    'model': {'btts': {'log_loss': 0.69429, 'base': 0.69349}, 'o15': {'log_loss': 0.61182, 'base': 0.61347}, 'o25': {'log_loss': 0.68187, 'base': 0.68722}, 'o35': {'log_loss': 0.55106, 'base': 0.55493}, 'exact': {'log_loss': 2.80308, 'base': 2.83359}},
                },
                'confirm': {
                    'market': {'btts': {'log_loss': 0.68837, 'base': 0.69292}, 'o15': {'log_loss': 0.62757, 'base': 0.63624}, 'o25': {'log_loss': 0.66403, 'base': 0.67693}, 'o35': {'log_loss': 0.49419, 'base': 0.50692}, 'exact': {'log_loss': 2.68528, 'base': 2.72903}},
                    'model': {'btts': {'log_loss': 0.68958, 'base': 0.69292}, 'o15': {'log_loss': 0.63158, 'base': 0.63624}, 'o25': {'log_loss': 0.66882, 'base': 0.67693}, 'o35': {'log_loss': 0.49847, 'base': 0.50692}, 'exact': {'log_loss': 2.69813, 'base': 2.72903}},
                },
            },
        },
    },
    'I2': {
        'name': 'Serie B',
        'country': 'Italy',
        'group': 'europe',
        'in_app': True,
        'pick': 'recalibrated',
        'model_alone': True,
        'numbers': {'market': ['o15', 'o25', 'o35', 'exact'], 'model': ['o15', 'o25', 'o35', 'exact']},
        'weights': {'season': 2026, 'w': [1.131, 0.0426, 0.0107]},
        'caveats': [
            'No shots in 2006-07 to 2016-17: the model rated it on goals alone in those seasons.',
            "Not shown from the market's prices: both teams to score - it did not beat the league's base rate in both 2006-2015 and 2016-2025.",
            "Not shown from the model alone: both teams to score - it did not beat the league's base rate in both 2006-2015 and 2016-2025.",
        ],
        'measured': {
            'price': 'pre-match',
            'season_check': {'last_result': '2026-10-28', 'days_since': 5, 'fixtures_posted': 10},
            'dev': {'seasons': [2006, 2015], 'matches': 4610, 'pick': 1.03042, 'pick_recalibrated': 1.03042, 'market': 1.03145, 'model': 1.05185, 'base': 1.07321, 'acc_pick': 0.4642, 'acc_market': 0.4651, 'acc_model': 0.4518},
            'confirm': {'seasons': [2016, 2025], 'matches': 3917, 'pick': 1.0452, 'pick_recalibrated': 1.0452, 'market': 1.04659, 'model': 1.05985, 'base': 1.08139, 'acc_pick': 0.448, 'acc_market': 0.4478, 'acc_model': 0.4381},
            'scores': {
                'dev': {
                    'market': {'btts': {'log_loss': 0.69884, 'base': 0.69324}, 'o15': {'log_loss': 0.60901, 'base': 0.61126}, 'o25': {'log_loss': 0.68095, 'base': 0.68698}, 'o35': {'log_loss': 0.53821, 'base': 0.54288}, 'exact': {'log_loss': 2.75073, 'base': 2.79458}},
                    'model': {'btts': {'log_loss': 0.69622, 'base': 0.69325}, 'o15': {'log_loss': 0.60967, 'base': 0.61112}, 'o25': {'log_loss': 0.6837, 'base': 0.68723}, 'o35': {'log_loss': 0.54018, 'base': 0.54375}, 'exact': {'log_loss': 2.76211, 'base': 2.7954}},
                },
                'confirm': {
                    'market': {'btts': {'log_loss': 0.69527, 'base': 0.69174}, 'o15': {'log_loss': 0.59005, 'base': 0.59385}, 'o25': {'log_loss': 0.68071, 'base': 0.6867}, 'o35': {'log_loss': 0.54132, 'base': 0.54584}, 'exact': {'log_loss': 2.75602, 'base': 2.79527}},
                    'model': {'btts': {'log_loss': 0.69424, 'base': 0.69174}, 'o15': {'log_loss': 0.59102, 'base': 0.59385}, 'o25': {'log_loss': 0.6832, 'base': 0.6867}, 'o35': {'log_loss': 0.54372, 'base': 0.54584}, 'exact': {'log_loss': 2.76768, 'base': 2.79527}},
                },
            },
        },
    },
    'D2': {
        'name': '2. Bundesliga',
        'country': 'Germany',
        'group': 'europe',
        'in_app': True,
        'pick': 'recalibrated',
        'model_alone': True,
        'numbers': {'market': ['btts', 'o15', 'o25', 'o35', 'exact'], 'model': ['btts', 'o15', 'o25', 'o35', 'exact']},
        'weights': {'season': 2026, 'w': [1.1311, 0.0426, 0.0107]},
        'caveats': [
            'No shots in 2007-08 to 2016-17: the model rated it on goals alone in those seasons.',
            'On 2016-2025 the recalibrated pick scored a little worse than the raw market (1.0497 against 1.0492).',
        ],
        'measured': {
            'price': 'pre-match',
            'season_check': {'last_result': '2026-10-25', 'days_since': 8, 'fixtures_posted': 9},
            'dev': {'seasons': [2006, 2015], 'matches': 3056, 'pick': 1.03428, 'pick_recalibrated': 1.03428, 'market': 1.03431, 'model': 1.04606, 'base': 1.07358, 'acc_pick': 0.4768, 'acc_market': 0.4771, 'acc_model': 0.462},
            'confirm': {'seasons': [2016, 2025], 'matches': 3060, 'pick': 1.04966, 'pick_recalibrated': 1.04966, 'market': 1.04923, 'model': 1.05494, 'base': 1.07755, 'acc_pick': 0.4618, 'acc_market': 0.4647, 'acc_model': 0.449},
            'scores': {
                'dev': {
                    'market': {'btts': {'log_loss': 0.69119, 'base': 0.69316}, 'o15': {'log_loss': 0.55343, 'base': 0.5595}, 'o25': {'log_loss': 0.68724, 'base': 0.69362}, 'o35': {'log_loss': 0.59416, 'base': 0.59923}, 'exact': {'log_loss': 2.90563, 'base': 2.96583}},
                    'model': {'btts': {'log_loss': 0.69221, 'base': 0.69316}, 'o15': {'log_loss': 0.55561, 'base': 0.5595}, 'o25': {'log_loss': 0.68967, 'base': 0.69362}, 'o35': {'log_loss': 0.5973, 'base': 0.59923}, 'exact': {'log_loss': 2.92049, 'base': 2.96583}},
                },
                'confirm': {
                    'market': {'btts': {'log_loss': 0.67875, 'base': 0.68555}, 'o15': {'log_loss': 0.50299, 'base': 0.51206}, 'o25': {'log_loss': 0.67784, 'base': 0.69025}, 'o35': {'log_loss': 0.62565, 'base': 0.63443}, 'exact': {'log_loss': 2.97015, 'base': 3.02347}},
                    'model': {'btts': {'log_loss': 0.68082, 'base': 0.68555}, 'o15': {'log_loss': 0.50767, 'base': 0.51206}, 'o25': {'log_loss': 0.68456, 'base': 0.69025}, 'o35': {'log_loss': 0.63108, 'base': 0.63443}, 'exact': {'log_loss': 2.98491, 'base': 3.02347}},
                },
            },
        },
    },
    'F2': {
        'name': 'Ligue 2',
        'country': 'France',
        'group': 'europe',
        'in_app': True,
        'pick': 'recalibrated',
        'model_alone': True,
        'numbers': {'market': ['o15', 'o25', 'o35', 'exact'], 'model': ['o25', 'o35', 'exact']},
        'weights': {'season': 2026, 'w': [1.1311, 0.0426, 0.0107]},
        'caveats': [
            'No shots in 2006-07 to 2016-17: the model rated it on goals alone in those seasons.',
            "Not shown from the market's prices: both teams to score - it did not beat the league's base rate in both 2006-2015 and 2016-2025.",
            "Not shown from the model alone: both teams to score, over 1.5 - it did not beat the league's base rate in both 2006-2015 and 2016-2025.",
        ],
        'measured': {
            'price': 'pre-match',
            'season_check': {'last_result': '2026-10-26', 'days_since': 7, 'fixtures_posted': 9},
            'dev': {'seasons': [2006, 2015], 'matches': 3790, 'pick': 1.03894, 'pick_recalibrated': 1.03894, 'market': 1.04015, 'model': 1.05203, 'base': 1.06212, 'acc_pick': 0.4612, 'acc_market': 0.4639, 'acc_model': 0.4607},
            'confirm': {'seasons': [2016, 2025], 'matches': 3550, 'pick': 1.04406, 'pick_recalibrated': 1.04406, 'market': 1.0446, 'model': 1.05524, 'base': 1.08876, 'acc_pick': 0.4606, 'acc_market': 0.4603, 'acc_model': 0.4518},
            'scores': {
                'dev': {
                    'market': {'btts': {'log_loss': 0.69488, 'base': 0.69371}, 'o15': {'log_loss': 0.62148, 'base': 0.62436}, 'o25': {'log_loss': 0.67487, 'base': 0.68064}, 'o35': {'log_loss': 0.51719, 'base': 0.52255}, 'exact': {'log_loss': 2.7094, 'base': 2.74173}},
                    'model': {'btts': {'log_loss': 0.69747, 'base': 0.69371}, 'o15': {'log_loss': 0.62591, 'base': 0.62436}, 'o25': {'log_loss': 0.6802, 'base': 0.68064}, 'o35': {'log_loss': 0.5205, 'base': 0.52255}, 'exact': {'log_loss': 2.72529, 'base': 2.74173}},
                },
                'confirm': {
                    'market': {'btts': {'log_loss': 0.68843, 'base': 0.69268}, 'o15': {'log_loss': 0.61988, 'base': 0.62309}, 'o25': {'log_loss': 0.67656, 'base': 0.68689}, 'o35': {'log_loss': 0.53693, 'base': 0.54722}, 'exact': {'log_loss': 2.79165, 'base': 2.85648}},
                    'model': {'btts': {'log_loss': 0.68931, 'base': 0.69268}, 'o15': {'log_loss': 0.62049, 'base': 0.62309}, 'o25': {'log_loss': 0.68049, 'base': 0.68689}, 'o35': {'log_loss': 0.54001, 'base': 0.54722}, 'exact': {'log_loss': 2.80522, 'base': 2.85648}},
                },
            },
        },
    },
    'E2': {
        'name': 'League One',
        'country': 'England',
        'group': 'europe',
        'in_app': True,
        'pick': 'recalibrated',
        'model_alone': True,
        'numbers': {'market': ['btts', 'o15', 'o25', 'o35', 'exact'], 'model': ['btts', 'o15', 'o25', 'o35', 'exact']},
        'weights': {'season': 2026, 'w': [1.1308, 0.0426, 0.0105]},
        'caveats': [
        ],
        'measured': {
            'price': 'pre-match',
            'season_check': {'last_result': '2026-10-27', 'days_since': 6, 'fixtures_posted': 12},
            'dev': {'seasons': [2006, 2015], 'matches': 5519, 'pick': 1.03944, 'pick_recalibrated': 1.03944, 'market': 1.04, 'model': 1.04826, 'base': 1.07583, 'acc_pick': 0.4722, 'acc_market': 0.4702, 'acc_model': 0.4657},
            'confirm': {'seasons': [2016, 2025], 'matches': 5367, 'pick': 1.02333, 'pick_recalibrated': 1.02333, 'market': 1.02469, 'model': 1.0398, 'base': 1.07456, 'acc_pick': 0.4919, 'acc_market': 0.4917, 'acc_model': 0.4761},
            'scores': {
                'dev': {
                    'market': {'btts': {'log_loss': 0.69081, 'base': 0.69247}, 'o15': {'log_loss': 0.57318, 'base': 0.57454}, 'o25': {'log_loss': 0.69003, 'base': 0.69333}, 'o35': {'log_loss': 0.57707, 'base': 0.58055}, 'exact': {'log_loss': 2.85958, 'base': 2.91181}},
                    'model': {'btts': {'log_loss': 0.6911, 'base': 0.69247}, 'o15': {'log_loss': 0.57327, 'base': 0.57454}, 'o25': {'log_loss': 0.69056, 'base': 0.69333}, 'o35': {'log_loss': 0.57761, 'base': 0.58055}, 'exact': {'log_loss': 2.87104, 'base': 2.91181}},
                },
                'confirm': {
                    'market': {'btts': {'log_loss': 0.6894, 'base': 0.69283}, 'o15': {'log_loss': 0.57295, 'base': 0.57764}, 'o25': {'log_loss': 0.6879, 'base': 0.69304}, 'o35': {'log_loss': 0.56736, 'base': 0.57265}, 'exact': {'log_loss': 2.84612, 'base': 2.91323}},
                    'model': {'btts': {'log_loss': 0.69058, 'base': 0.69283}, 'o15': {'log_loss': 0.57466, 'base': 0.57764}, 'o25': {'log_loss': 0.69007, 'base': 0.69304}, 'o35': {'log_loss': 0.56882, 'base': 0.57265}, 'exact': {'log_loss': 2.8673, 'base': 2.91323}},
                },
            },
        },
    },
    'E3': {
        'name': 'League Two',
        'country': 'England',
        'group': 'europe',
        'in_app': True,
        'pick': 'raw',
        'model_alone': True,
        'numbers': {'market': ['btts', 'o15', 'o25', 'o35', 'exact'], 'model': ['btts', 'o15', 'o25', 'o35', 'exact']},
        'weights': None,
        'caveats': [
            'The recalibrated market lost to the raw market on 2006-2015 (1.0577 against 1.0573), so its pick is the raw no-vig market.',
        ],
        'measured': {
            'price': 'pre-match',
            'season_check': {'last_result': '2026-10-27', 'days_since': 6, 'fixtures_posted': 12},
            'dev': {'seasons': [2006, 2015], 'matches': 5519, 'pick': 1.05727, 'pick_recalibrated': 1.05767, 'market': 1.05727, 'model': 1.0609, 'base': 1.08417, 'acc_pick': 0.4535, 'acc_market': 0.4535, 'acc_model': 0.4446},
            'confirm': {'seasons': [2016, 2025], 'matches': 5403, 'pick': 1.04907, 'pick_recalibrated': 1.04897, 'market': 1.04907, 'model': 1.05834, 'base': 1.07982, 'acc_pick': 0.4603, 'acc_market': 0.4603, 'acc_model': 0.446},
            'scores': {
                'dev': {
                    'market': {'btts': {'log_loss': 0.69189, 'base': 0.69348}, 'o15': {'log_loss': 0.58616, 'base': 0.58828}, 'o25': {'log_loss': 0.69039, 'base': 0.69184}, 'o35': {'log_loss': 0.55818, 'base': 0.56017}, 'exact': {'log_loss': 2.84451, 'base': 2.88764}},
                    'model': {'btts': {'log_loss': 0.69288, 'base': 0.69348}, 'o15': {'log_loss': 0.58744, 'base': 0.58828}, 'o25': {'log_loss': 0.69087, 'base': 0.69184}, 'o35': {'log_loss': 0.55858, 'base': 0.56017}, 'exact': {'log_loss': 2.85162, 'base': 2.88764}},
                },
                'confirm': {
                    'market': {'btts': {'log_loss': 0.68957, 'base': 0.69276}, 'o15': {'log_loss': 0.58639, 'base': 0.59019}, 'o25': {'log_loss': 0.68618, 'base': 0.69198}, 'o35': {'log_loss': 0.56477, 'base': 0.56971}, 'exact': {'log_loss': 2.83952, 'base': 2.88737}},
                    'model': {'btts': {'log_loss': 0.69048, 'base': 0.69276}, 'o15': {'log_loss': 0.58718, 'base': 0.59019}, 'o25': {'log_loss': 0.68746, 'base': 0.69198}, 'o35': {'log_loss': 0.56529, 'base': 0.56971}, 'exact': {'log_loss': 2.85293, 'base': 2.88737}},
                },
            },
        },
    },
    'EC': {
        'name': 'National League',
        'country': 'England',
        'group': 'europe',
        'in_app': True,
        'pick': 'recalibrated',
        'model_alone': True,
        'numbers': {'market': ['btts', 'o15', 'o25', 'o35', 'exact'], 'model': ['btts', 'o15', 'o25', 'o35', 'exact']},
        'weights': {'season': 2026, 'w': [1.1311, 0.0426, 0.0107]},
        'caveats': [
            'No shots in 2016-17 to 2025-26: the model rated it on goals alone in those seasons.',
        ],
        'measured': {
            'price': 'pre-match',
            'season_check': {'last_result': '2026-10-28', 'days_since': 5, 'fixtures_posted': 12},
            'dev': {'seasons': [2006, 2015], 'matches': 5437, 'pick': 1.02552, 'pick_recalibrated': 1.02552, 'market': 1.02601, 'model': 1.03353, 'base': 1.07308, 'acc_pick': 0.4922, 'acc_market': 0.4909, 'acc_model': 0.4828},
            'confirm': {'seasons': [2016, 2025], 'matches': 5284, 'pick': 1.02432, 'pick_recalibrated': 1.02432, 'market': 1.02476, 'model': 1.03343, 'base': 1.07727, 'acc_pick': 0.482, 'acc_market': 0.4839, 'acc_model': 0.4748},
            'scores': {
                'dev': {
                    'market': {'btts': {'log_loss': 0.68823, 'base': 0.69195}, 'o15': {'log_loss': 0.57055, 'base': 0.57393}, 'o25': {'log_loss': 0.68962, 'base': 0.69379}, 'o35': {'log_loss': 0.59976, 'base': 0.60383}, 'exact': {'log_loss': 2.89225, 'base': 2.96781}},
                    'model': {'btts': {'log_loss': 0.68981, 'base': 0.69194}, 'o15': {'log_loss': 0.57163, 'base': 0.57425}, 'o25': {'log_loss': 0.69102, 'base': 0.69379}, 'o35': {'log_loss': 0.60214, 'base': 0.60439}, 'exact': {'log_loss': 2.90468, 'base': 2.96868}},
                },
                'confirm': {
                    'market': {'btts': {'log_loss': 0.68977, 'base': 0.69138}, 'o15': {'log_loss': 0.55465, 'base': 0.55934}, 'o25': {'log_loss': 0.68659, 'base': 0.69312}, 'o35': {'log_loss': 0.59887, 'base': 0.60505}, 'exact': {'log_loss': 2.906, 'base': 2.97704}},
                    'model': {'btts': {'log_loss': 0.68958, 'base': 0.69138}, 'o15': {'log_loss': 0.55545, 'base': 0.55934}, 'o25': {'log_loss': 0.68741, 'base': 0.69312}, 'o35': {'log_loss': 0.59969, 'base': 0.60505}, 'exact': {'log_loss': 2.91757, 'base': 2.97704}},
                },
            },
        },
    },
    'SC0': {
        'name': 'Premiership',
        'country': 'Scotland',
        'group': 'europe',
        'in_app': True,
        'pick': 'raw',
        'model_alone': True,
        'numbers': {'market': ['btts', 'o15', 'o25', 'o35', 'exact'], 'model': ['btts', 'o15', 'o25', 'o35', 'exact']},
        'weights': None,
        'caveats': [
            'The recalibrated market lost to the raw market on 2006-2015 (0.9844 against 0.9844), so its pick is the raw no-vig market.',
        ],
        'measured': {
            'price': 'pre-match',
            'season_check': {'last_result': '2026-10-28', 'days_since': 5, 'fixtures_posted': 3},
            'dev': {'seasons': [2006, 2015], 'matches': 2275, 'pick': 0.98436, 'pick_recalibrated': 0.98441, 'market': 0.98436, 'model': 0.99403, 'base': 1.07318, 'acc_pick': 0.513, 'acc_market': 0.513, 'acc_model': 0.5112},
            'confirm': {'seasons': [2016, 2025], 'matches': 2231, 'pick': 0.93839, 'pick_recalibrated': 0.93686, 'market': 0.93839, 'model': 0.95171, 'base': 1.06902, 'acc_pick': 0.5522, 'acc_market': 0.5522, 'acc_model': 0.5459},
            'scores': {
                'dev': {
                    'market': {'btts': {'log_loss': 0.68938, 'base': 0.69375}, 'o15': {'log_loss': 0.57502, 'base': 0.57979}, 'o25': {'log_loss': 0.68681, 'base': 0.69467}, 'o35': {'log_loss': 0.5634, 'base': 0.5716}, 'exact': {'log_loss': 2.83605, 'base': 2.96691}},
                    'model': {'btts': {'log_loss': 0.69051, 'base': 0.69375}, 'o15': {'log_loss': 0.57619, 'base': 0.57979}, 'o25': {'log_loss': 0.68894, 'base': 0.69467}, 'o35': {'log_loss': 0.56402, 'base': 0.5716}, 'exact': {'log_loss': 2.84861, 'base': 2.96691}},
                },
                'confirm': {
                    'market': {'btts': {'log_loss': 0.68543, 'base': 0.69325}, 'o15': {'log_loss': 0.54217, 'base': 0.55377}, 'o25': {'log_loss': 0.68033, 'base': 0.69336}, 'o35': {'log_loss': 0.57894, 'base': 0.59435}, 'exact': {'log_loss': 2.79882, 'base': 2.97639}},
                    'model': {'btts': {'log_loss': 0.69056, 'base': 0.69325}, 'o15': {'log_loss': 0.54676, 'base': 0.55377}, 'o25': {'log_loss': 0.6869, 'base': 0.69336}, 'o35': {'log_loss': 0.58515, 'base': 0.59435}, 'exact': {'log_loss': 2.82195, 'base': 2.97639}},
                },
            },
        },
    },
    'SC1': {
        'name': 'Championship',
        'country': 'Scotland',
        'group': 'europe',
        'in_app': True,
        'pick': 'raw',
        'model_alone': True,
        'numbers': {'market': ['btts', 'o15', 'o25', 'o35', 'exact'], 'model': ['o15', 'o25', 'o35', 'exact']},
        'weights': None,
        'caveats': [
            'No shots in 2006-07 to 2016-17: the model rated it on goals alone in those seasons.',
            'The recalibrated market lost to the raw market on 2006-2015 (1.0216 against 1.0207), so its pick is the raw no-vig market.',
            "Not shown from the model alone: both teams to score - it did not beat the league's base rate in both 2006-2015 and 2016-2025.",
        ],
        'measured': {
            'price': 'pre-match',
            'season_check': {'last_result': '2026-10-27', 'days_since': 6, 'fixtures_posted': 5},
            'dev': {'seasons': [2006, 2015], 'matches': 1799, 'pick': 1.02068, 'pick_recalibrated': 1.02161, 'market': 1.02068, 'model': 1.02758, 'base': 1.07925, 'acc_pick': 0.4964, 'acc_market': 0.4964, 'acc_model': 0.483},
            'confirm': {'seasons': [2016, 2025], 'matches': 1712, 'pick': 1.04189, 'pick_recalibrated': 1.0431, 'market': 1.04189, 'model': 1.05173, 'base': 1.08854, 'acc_pick': 0.4562, 'acc_market': 0.4562, 'acc_model': 0.4504},
            'scores': {
                'dev': {
                    'market': {'btts': {'log_loss': 0.69291, 'base': 0.6936}, 'o15': {'log_loss': 0.56568, 'base': 0.56889}, 'o25': {'log_loss': 0.68795, 'base': 0.6947}, 'o35': {'log_loss': 0.60146, 'base': 0.60801}, 'exact': {'log_loss': 2.93559, 'base': 3.03449}},
                    'model': {'btts': {'log_loss': 0.69027, 'base': 0.6936}, 'o15': {'log_loss': 0.56536, 'base': 0.56889}, 'o25': {'log_loss': 0.68556, 'base': 0.6947}, 'o35': {'log_loss': 0.60268, 'base': 0.60801}, 'exact': {'log_loss': 2.94505, 'base': 3.03449}},
                },
                'confirm': {
                    'market': {'btts': {'log_loss': 0.69287, 'base': 0.69343}, 'o15': {'log_loss': 0.5836, 'base': 0.58468}, 'o25': {'log_loss': 0.6892, 'base': 0.69325}, 'o35': {'log_loss': 0.57874, 'base': 0.58485}, 'exact': {'log_loss': 2.89063, 'base': 2.95113}},
                    'model': {'btts': {'log_loss': 0.69526, 'base': 0.69343}, 'o15': {'log_loss': 0.58413, 'base': 0.58468}, 'o25': {'log_loss': 0.69012, 'base': 0.69325}, 'o35': {'log_loss': 0.58183, 'base': 0.58485}, 'exact': {'log_loss': 2.90492, 'base': 2.95113}},
                },
            },
        },
    },
    'SC2': {
        'name': 'League One',
        'country': 'Scotland',
        'group': 'europe',
        'in_app': True,
        'pick': 'recalibrated',
        'model_alone': True,
        'numbers': {'market': ['btts', 'o15', 'exact'], 'model': ['btts', 'o15', 'o25', 'o35', 'exact']},
        'weights': {'season': 2026, 'w': [1.1311, 0.0426, 0.0106]},
        'caveats': [
            'No shots in 2006-07 to 2016-17: the model rated it on goals alone in those seasons.',
            "Not shown from the market's prices: over 2.5, over 3.5 - it did not beat the league's base rate in both 2006-2015 and 2016-2025.",
        ],
        'measured': {
            'price': 'pre-match',
            'season_check': {'last_result': '2026-10-27', 'days_since': 6, 'fixtures_posted': 5},
            'dev': {'seasons': [2006, 2015], 'matches': 1798, 'pick': 1.00137, 'pick_recalibrated': 1.00137, 'market': 1.00363, 'model': 1.01626, 'base': 1.06059, 'acc_pick': 0.5089, 'acc_market': 0.5067, 'acc_model': 0.495},
            'confirm': {'seasons': [2016, 2025], 'matches': 1689, 'pick': 1.01058, 'pick_recalibrated': 1.01058, 'market': 1.01129, 'model': 1.01764, 'base': 1.07198, 'acc_pick': 0.5074, 'acc_market': 0.5068, 'acc_model': 0.5003},
            'scores': {
                'dev': {
                    'market': {'btts': {'log_loss': 0.68329, 'base': 0.68349}, 'o15': {'log_loss': 0.49906, 'base': 0.50011}, 'o25': {'log_loss': 0.6838, 'base': 0.6807}, 'o35': {'log_loss': 0.66198, 'base': 0.65843}, 'exact': {'log_loss': 3.03242, 'base': 3.11798}},
                    'model': {'btts': {'log_loss': 0.68198, 'base': 0.68346}, 'o15': {'log_loss': 0.49411, 'base': 0.49968}, 'o25': {'log_loss': 0.67539, 'base': 0.68066}, 'o35': {'log_loss': 0.65524, 'base': 0.65874}, 'exact': {'log_loss': 3.04278, 'base': 3.11896}},
                },
                'confirm': {
                    'market': {'btts': {'log_loss': 0.68621, 'base': 0.69056}, 'o15': {'log_loss': 0.50611, 'base': 0.50738}, 'o25': {'log_loss': 0.68397, 'base': 0.68984}, 'o35': {'log_loss': 0.62278, 'base': 0.62985}, 'exact': {'log_loss': 2.95598, 'base': 3.05317}},
                    'model': {'btts': {'log_loss': 0.68993, 'base': 0.69056}, 'o15': {'log_loss': 0.50599, 'base': 0.50738}, 'o25': {'log_loss': 0.68548, 'base': 0.68984}, 'o35': {'log_loss': 0.62457, 'base': 0.62985}, 'exact': {'log_loss': 2.96998, 'base': 3.05317}},
                },
            },
        },
    },
    'SC3': {
        'name': 'League Two',
        'country': 'Scotland',
        'group': 'europe',
        'in_app': True,
        'pick': 'raw',
        'model_alone': True,
        'numbers': {'market': ['btts', 'o15', 'o25', 'o35', 'exact'], 'model': ['btts', 'o15', 'o25', 'o35', 'exact']},
        'weights': None,
        'caveats': [
            'No shots in 2006-07 to 2016-17: the model rated it on goals alone in those seasons.',
            'The recalibrated market lost to the raw market on 2006-2015 (1.0283 against 1.0268), so its pick is the raw no-vig market.',
        ],
        'measured': {
            'price': 'pre-match',
            'season_check': {'last_result': '2026-10-17', 'days_since': 16, 'fixtures_posted': 5},
            'dev': {'seasons': [2006, 2015], 'matches': 1797, 'pick': 1.02682, 'pick_recalibrated': 1.02827, 'market': 1.02682, 'model': 1.02987, 'base': 1.05535, 'acc_pick': 0.4925, 'acc_market': 0.4925, 'acc_model': 0.4891},
            'confirm': {'seasons': [2016, 2025], 'matches': 1687, 'pick': 1.02194, 'pick_recalibrated': 1.02155, 'market': 1.02194, 'model': 1.04096, 'base': 1.07575, 'acc_pick': 0.4849, 'acc_market': 0.4849, 'acc_model': 0.4707},
            'scores': {
                'dev': {
                    'market': {'btts': {'log_loss': 0.68594, 'base': 0.69222}, 'o15': {'log_loss': 0.50164, 'base': 0.50607}, 'o25': {'log_loss': 0.6834, 'base': 0.68947}, 'o35': {'log_loss': 0.62815, 'base': 0.63652}, 'exact': {'log_loss': 3.03097, 'base': 3.10121}},
                    'model': {'btts': {'log_loss': 0.68539, 'base': 0.69222}, 'o15': {'log_loss': 0.4992, 'base': 0.50648}, 'o25': {'log_loss': 0.68048, 'base': 0.68947}, 'o35': {'log_loss': 0.62317, 'base': 0.6367}, 'exact': {'log_loss': 3.03849, 'base': 3.10153}},
                },
                'confirm': {
                    'market': {'btts': {'log_loss': 0.68983, 'base': 0.69246}, 'o15': {'log_loss': 0.53913, 'base': 0.54356}, 'o25': {'log_loss': 0.68878, 'base': 0.69369}, 'o35': {'log_loss': 0.61021, 'base': 0.61339}, 'exact': {'log_loss': 2.93513, 'base': 3.00914}},
                    'model': {'btts': {'log_loss': 0.69042, 'base': 0.69246}, 'o15': {'log_loss': 0.53854, 'base': 0.54356}, 'o25': {'log_loss': 0.68771, 'base': 0.69369}, 'o35': {'log_loss': 0.60892, 'base': 0.61339}, 'exact': {'log_loss': 2.96048, 'base': 3.00914}},
                },
            },
        },
    },
    'N1': {
        'name': 'Eredivisie',
        'country': 'Netherlands',
        'group': 'europe',
        'in_app': True,
        'pick': 'recalibrated',
        'model_alone': True,
        'numbers': {'market': ['btts', 'o15', 'o25', 'o35', 'exact'], 'model': ['btts', 'o15', 'o25', 'o35', 'exact']},
        'weights': {'season': 2026, 'w': [1.1311, 0.0426, 0.0107]},
        'caveats': [
            'No shots in 2006-07 to 2016-17: the model rated it on goals alone in those seasons.',
        ],
        'measured': {
            'price': 'pre-match',
            'season_check': {'last_result': '2026-10-25', 'days_since': 8, 'fixtures_posted': 9},
            'dev': {'seasons': [2006, 2015], 'matches': 3060, 'pick': 0.94841, 'pick_recalibrated': 0.94841, 'market': 0.94895, 'model': 0.9567, 'base': 1.05165, 'acc_pick': 0.5575, 'acc_market': 0.5565, 'acc_model': 0.5493},
            'confirm': {'seasons': [2016, 2025], 'matches': 2986, 'pick': 0.93471, 'pick_recalibrated': 0.93471, 'market': 0.93488, 'model': 0.94399, 'base': 1.06376, 'acc_pick': 0.564, 'acc_market': 0.5636, 'acc_model': 0.5539},
            'scores': {
                'dev': {
                    'market': {'btts': {'log_loss': 0.68192, 'base': 0.68755}, 'o15': {'log_loss': 0.46384, 'base': 0.46982}, 'o25': {'log_loss': 0.67379, 'base': 0.68099}, 'o35': {'log_loss': 0.65692, 'base': 0.66446}, 'exact': {'log_loss': 2.98108, 'base': 3.11148}},
                    'model': {'btts': {'log_loss': 0.68044, 'base': 0.68755}, 'o15': {'log_loss': 0.46553, 'base': 0.46982}, 'o25': {'log_loss': 0.67495, 'base': 0.68099}, 'o35': {'log_loss': 0.65826, 'base': 0.66446}, 'exact': {'log_loss': 2.99385, 'base': 3.11148}},
                },
                'confirm': {
                    'market': {'btts': {'log_loss': 0.68117, 'base': 0.68485}, 'o15': {'log_loss': 0.45769, 'base': 0.46678}, 'o25': {'log_loss': 0.65912, 'base': 0.67525}, 'o35': {'log_loss': 0.64621, 'base': 0.66317}, 'exact': {'log_loss': 2.95477, 'base': 3.1266}},
                    'model': {'btts': {'log_loss': 0.68132, 'base': 0.68485}, 'o15': {'log_loss': 0.45984, 'base': 0.46678}, 'o25': {'log_loss': 0.66321, 'base': 0.67525}, 'o35': {'log_loss': 0.64934, 'base': 0.66317}, 'exact': {'log_loss': 2.96751, 'base': 3.1266}},
                },
            },
        },
    },
    'B1': {
        'name': 'Pro League',
        'country': 'Belgium',
        'group': 'europe',
        'in_app': True,
        'pick': 'recalibrated',
        'model_alone': True,
        'numbers': {'market': ['o15', 'o25', 'o35', 'exact'], 'model': ['btts', 'o15', 'o25', 'o35', 'exact']},
        'weights': {'season': 2026, 'w': [1.1311, 0.0426, 0.0107]},
        'caveats': [
            'No shots in 2006-07 to 2016-17: the model rated it on goals alone in those seasons.',
            "Not shown from the market's prices: both teams to score - it did not beat the league's base rate in both 2006-2015 and 2016-2025.",
        ],
        'measured': {
            'price': 'pre-match',
            'season_check': {'last_result': '2026-10-25', 'days_since': 8, 'fixtures_posted': 9},
            'dev': {'seasons': [2006, 2015], 'matches': 2568, 'pick': 0.97913, 'pick_recalibrated': 0.97913, 'market': 0.9807, 'model': 0.98791, 'base': 1.05703, 'acc_pick': 0.53, 'acc_market': 0.5319, 'acc_model': 0.5257},
            'confirm': {'seasons': [2016, 2025], 'matches': 2805, 'pick': 0.99036, 'pick_recalibrated': 0.99036, 'market': 0.99101, 'model': 1.00371, 'base': 1.07315, 'acc_pick': 0.5205, 'acc_market': 0.5201, 'acc_model': 0.5105},
            'scores': {
                'dev': {
                    'market': {'btts': {'log_loss': 0.68689, 'base': 0.69182}, 'o15': {'log_loss': 0.53772, 'base': 0.54306}, 'o25': {'log_loss': 0.68877, 'base': 0.69402}, 'o35': {'log_loss': 0.61415, 'base': 0.62067}, 'exact': {'log_loss': 2.87697, 'base': 2.98583}},
                    'model': {'btts': {'log_loss': 0.69048, 'base': 0.69182}, 'o15': {'log_loss': 0.53926, 'base': 0.54306}, 'o25': {'log_loss': 0.68911, 'base': 0.69402}, 'o35': {'log_loss': 0.61621, 'base': 0.62067}, 'exact': {'log_loss': 2.89073, 'base': 2.98583}},
                },
                'confirm': {
                    'market': {'btts': {'log_loss': 0.68769, 'base': 0.68734}, 'o15': {'log_loss': 0.51061, 'base': 0.51964}, 'o25': {'log_loss': 0.67795, 'base': 0.68953}, 'o35': {'log_loss': 0.61905, 'base': 0.62857}, 'exact': {'log_loss': 2.91513, 'base': 3.02231}},
                    'model': {'btts': {'log_loss': 0.6869, 'base': 0.68734}, 'o15': {'log_loss': 0.51476, 'base': 0.51964}, 'o25': {'log_loss': 0.68192, 'base': 0.68953}, 'o35': {'log_loss': 0.62092, 'base': 0.62857}, 'exact': {'log_loss': 2.93115, 'base': 3.02231}},
                },
            },
        },
    },
    'P1': {
        'name': 'Primeira Liga',
        'country': 'Portugal',
        'group': 'europe',
        'in_app': True,
        'pick': 'recalibrated',
        'model_alone': True,
        'numbers': {'market': ['btts', 'o15', 'o25', 'o35', 'exact'], 'model': ['btts', 'o15', 'o25', 'o35', 'exact']},
        'weights': {'season': 2026, 'w': [1.1311, 0.0426, 0.0107]},
        'caveats': [
            'No shots in 2006-07 to 2016-17: the model rated it on goals alone in those seasons.',
        ],
        'measured': {
            'price': 'pre-match',
            'season_check': {'last_result': '2026-10-26', 'days_since': 7, 'fixtures_posted': 9},
            'dev': {'seasons': [2006, 2015], 'matches': 2528, 'pick': 0.944, 'pick_recalibrated': 0.944, 'market': 0.94777, 'model': 0.95509, 'base': 1.07318, 'acc_pick': 0.5411, 'acc_market': 0.54, 'acc_model': 0.536},
            'confirm': {'seasons': [2016, 2025], 'matches': 3060, 'pick': 0.91652, 'pick_recalibrated': 0.91652, 'market': 0.9205, 'model': 0.93485, 'base': 1.06951, 'acc_pick': 0.5667, 'acc_market': 0.5657, 'acc_model': 0.5542},
            'scores': {
                'dev': {
                    'market': {'btts': {'log_loss': 0.68989, 'base': 0.69356}, 'o15': {'log_loss': 0.60361, 'base': 0.61489}, 'o25': {'log_loss': 0.67797, 'base': 0.69091}, 'o35': {'log_loss': 0.56439, 'base': 0.5746}, 'exact': {'log_loss': 2.73551, 'base': 2.91417}},
                    'model': {'btts': {'log_loss': 0.69034, 'base': 0.6935}, 'o15': {'log_loss': 0.60388, 'base': 0.61499}, 'o25': {'log_loss': 0.67901, 'base': 0.69081}, 'o35': {'log_loss': 0.56669, 'base': 0.57435}, 'exact': {'log_loss': 2.74701, 'base': 2.91364}},
                },
                'confirm': {
                    'market': {'btts': {'log_loss': 0.69044, 'base': 0.69295}, 'o15': {'log_loss': 0.57574, 'base': 0.59047}, 'o25': {'log_loss': 0.67529, 'base': 0.69468}, 'o35': {'log_loss': 0.55972, 'base': 0.57637}, 'exact': {'log_loss': 2.74478, 'base': 2.93632}},
                    'model': {'btts': {'log_loss': 0.68969, 'base': 0.69295}, 'o15': {'log_loss': 0.5792, 'base': 0.59047}, 'o25': {'log_loss': 0.68065, 'base': 0.69468}, 'o35': {'log_loss': 0.56315, 'base': 0.57637}, 'exact': {'log_loss': 2.76042, 'base': 2.93632}},
                },
            },
        },
    },
    'T1': {
        'name': 'Süper Lig',
        'country': 'Turkey',
        'group': 'europe',
        'in_app': True,
        'pick': 'recalibrated',
        'model_alone': True,
        'numbers': {'market': ['o15', 'o25', 'o35', 'exact'], 'model': ['o15', 'o25', 'o35', 'exact']},
        'weights': {'season': 2026, 'w': [1.1308, 0.0425, 0.0105]},
        'caveats': [
            'No shots in 2006-07 to 2016-17: the model rated it on goals alone in those seasons.',
            "Not shown from the market's prices: both teams to score - it did not beat the league's base rate in both 2006-2015 and 2016-2025.",
            "Not shown from the model alone: both teams to score - it did not beat the league's base rate in both 2006-2015 and 2016-2025.",
        ],
        'measured': {
            'price': 'pre-match',
            'season_check': {'last_result': '2026-10-26', 'days_since': 7, 'fixtures_posted': 9},
            'dev': {'seasons': [2006, 2015], 'matches': 3023, 'pick': 0.99756, 'pick_recalibrated': 0.99756, 'market': 0.99828, 'model': 1.00952, 'base': 1.06414, 'acc_pick': 0.5207, 'acc_market': 0.5197, 'acc_model': 0.5114},
            'confirm': {'seasons': [2016, 2025], 'matches': 3365, 'pick': 0.97777, 'pick_recalibrated': 0.97777, 'market': 0.98065, 'model': 0.99636, 'base': 1.06724, 'acc_pick': 0.5311, 'acc_market': 0.5314, 'acc_model': 0.5224},
            'scores': {
                'dev': {
                    'market': {'btts': {'log_loss': 0.69433, 'base': 0.69259}, 'o15': {'log_loss': 0.58302, 'base': 0.58882}, 'o25': {'log_loss': 0.68769, 'base': 0.69411}, 'o35': {'log_loss': 0.58475, 'base': 0.59055}, 'exact': {'log_loss': 2.85432, 'base': 2.93398}},
                    'model': {'btts': {'log_loss': 0.69374, 'base': 0.6926}, 'o15': {'log_loss': 0.58434, 'base': 0.58874}, 'o25': {'log_loss': 0.6903, 'base': 0.69412}, 'o35': {'log_loss': 0.58758, 'base': 0.59046}, 'exact': {'log_loss': 2.86356, 'base': 2.93412}},
                },
                'confirm': {
                    'market': {'btts': {'log_loss': 0.6868, 'base': 0.68769}, 'o15': {'log_loss': 0.529, 'base': 0.54133}, 'o25': {'log_loss': 0.67411, 'base': 0.6923}, 'o35': {'log_loss': 0.60831, 'base': 0.62246}, 'exact': {'log_loss': 2.89151, 'base': 3.00754}},
                    'model': {'btts': {'log_loss': 0.68797, 'base': 0.68769}, 'o15': {'log_loss': 0.53294, 'base': 0.54133}, 'o25': {'log_loss': 0.68012, 'base': 0.6923}, 'o35': {'log_loss': 0.61143, 'base': 0.62246}, 'exact': {'log_loss': 2.91403, 'base': 3.00754}},
                },
            },
        },
    },
    'G1': {
        'name': 'Super League',
        'country': 'Greece',
        'group': 'europe',
        'in_app': True,
        'pick': 'recalibrated',
        'model_alone': True,
        'numbers': {'market': ['o15', 'o25', 'o35', 'exact'], 'model': ['btts', 'o15', 'o25', 'o35', 'exact']},
        'weights': {'season': 2026, 'w': [1.131, 0.0426, 0.0106]},
        'caveats': [
            'No shots in 2006-07 to 2016-17: the model rated it on goals alone in those seasons.',
            "Not shown from the market's prices: both teams to score - it did not beat the league's base rate in both 2006-2015 and 2016-2025.",
        ],
        'measured': {
            'price': 'pre-match',
            'season_check': {'last_result': '2026-10-25', 'days_since': 8, 'fixtures_posted': 7},
            'dev': {'seasons': [2006, 2015], 'matches': 2502, 'pick': 0.92854, 'pick_recalibrated': 0.92854, 'market': 0.93252, 'model': 0.95113, 'base': 1.04712, 'acc_pick': 0.5544, 'acc_market': 0.5548, 'acc_model': 0.5504},
            'confirm': {'seasons': [2016, 2025], 'matches': 2387, 'pick': 0.93029, 'pick_recalibrated': 0.93029, 'market': 0.93277, 'model': 0.95299, 'base': 1.08139, 'acc_pick': 0.5337, 'acc_market': 0.5341, 'acc_model': 0.5291},
            'scores': {
                'dev': {
                    'market': {'btts': {'log_loss': 0.68099, 'base': 0.67995}, 'o15': {'log_loss': 0.63064, 'base': 0.63974}, 'o25': {'log_loss': 0.66156, 'base': 0.67664}, 'o35': {'log_loss': 0.49332, 'base': 0.5079}, 'exact': {'log_loss': 2.65018, 'base': 2.80201}},
                    'model': {'btts': {'log_loss': 0.67809, 'base': 0.67993}, 'o15': {'log_loss': 0.63221, 'base': 0.64006}, 'o25': {'log_loss': 0.66338, 'base': 0.677}, 'o35': {'log_loss': 0.49484, 'base': 0.50988}, 'exact': {'log_loss': 2.66726, 'base': 2.8047}},
                },
                'confirm': {
                    'market': {'btts': {'log_loss': 0.67686, 'base': 0.68726}, 'o15': {'log_loss': 0.59731, 'base': 0.61086}, 'o25': {'log_loss': 0.66923, 'base': 0.68836}, 'o35': {'log_loss': 0.52081, 'base': 0.53316}, 'exact': {'log_loss': 2.66156, 'base': 2.8672}},
                    'model': {'btts': {'log_loss': 0.68176, 'base': 0.68726}, 'o15': {'log_loss': 0.60046, 'base': 0.61086}, 'o25': {'log_loss': 0.67542, 'base': 0.68836}, 'o35': {'log_loss': 0.52437, 'base': 0.53316}, 'exact': {'log_loss': 2.68926, 'base': 2.8672}},
                },
            },
        },
    },
    'ARG': {
        'name': 'Liga Profesional',
        'country': 'Argentina',
        'group': 'extra',
        'in_app': True,
        'pick': 'recalibrated',
        'model_alone': True,
        'numbers': {'market': None, 'model': ['btts', 'o15', 'exact']},
        'weights': {'season': 2026, 'w': [1.106, 0.0297, -0.0015]},
        'caveats': [
            "Measured on closing prices, the only ones its history has; its live pick uses the pre-match price. On the European divisions' 2019-2025, where both exist, the pre-match price scored 0.0035 worse in log loss than the closing one (1.0026 against 0.9990, 53,270 matches; accuracy 50.4% against 50.7%): its live picks should do about that much worse than measured, and its band rates were measured on closing-price picks.",
            'No shots in its history: the model rates its clubs on goals alone.',
            "No over/under price: every goals and scores number is the model's own.",
            'On 2019-2025 the recalibrated pick scored a little worse than the raw market (1.0538 against 1.0529).',
            "Not shown from the model alone: over 2.5, over 3.5 - it did not beat the league's base rate in both 2013-2018 and 2019-2025.",
        ],
        'measured': {
            'price': 'closing',
            'season_check': {'last_result': '2026-10-27', 'days_since': 6, 'fixtures_posted': 15},
            'dev': {'seasons': [2013, 2018], 'matches': 2435, 'pick': 1.02132, 'pick_recalibrated': 1.02132, 'market': 1.02248, 'model': 1.04391, 'base': 1.07323, 'acc_pick': 0.4854, 'acc_market': 0.485, 'acc_model': 0.4698},
            'confirm': {'seasons': [2019, 2025], 'matches': 3165, 'pick': 1.05382, 'pick_recalibrated': 1.05382, 'market': 1.05287, 'model': 1.06842, 'base': 1.08069, 'acc_pick': 0.4468, 'acc_market': 0.4455, 'acc_model': 0.4303},
            'scores': {
                'dev': {
                    'market': {},
                    'model': {'btts': {'log_loss': 0.68842, 'base': 0.68893}, 'o15': {'log_loss': 0.63766, 'base': 0.63802}, 'o25': {'log_loss': 0.6779, 'base': 0.67639}, 'o35': {'log_loss': 0.5058, 'base': 0.50439}, 'exact': {'log_loss': 2.73149, 'base': 2.77455}},
                },
                'confirm': {
                    'market': {},
                    'model': {'btts': {'log_loss': 0.68231, 'base': 0.68653}, 'o15': {'log_loss': 0.64875, 'base': 0.65383}, 'o25': {'log_loss': 0.65645, 'base': 0.66375}, 'o35': {'log_loss': 0.47444, 'base': 0.48019}, 'exact': {'log_loss': 2.6955, 'base': 2.72473}},
                },
            },
        },
    },
    'AUT': {
        'name': 'Bundesliga',
        'country': 'Austria',
        'group': 'extra',
        'in_app': True,
        'pick': 'raw',
        'model_alone': True,
        'numbers': {'market': None, 'model': ['btts', 'o15', 'o25', 'o35', 'exact']},
        'weights': None,
        'caveats': [
            "Measured on closing prices, the only ones its history has; its live pick uses the pre-match price. On the European divisions' 2019-2025, where both exist, the pre-match price scored 0.0035 worse in log loss than the closing one (1.0026 against 0.9990, 53,270 matches; accuracy 50.4% against 50.7%): its live picks should do about that much worse than measured, and its band rates were measured on closing-price picks.",
            'No shots in its history: the model rates its clubs on goals alone.',
            "No over/under price: every goals and scores number is the model's own.",
            'The recalibrated market lost to the raw market on 2013-2018 (0.9991 against 0.9969), so its pick is the raw no-vig market.',
        ],
        'measured': {
            'price': 'closing',
            'season_check': {'last_result': '2026-10-25', 'days_since': 8, 'fixtures_posted': 6},
            'dev': {'seasons': [2013, 2018], 'matches': 1097, 'pick': 0.99689, 'pick_recalibrated': 0.99914, 'market': 0.99689, 'model': 1.01224, 'base': 1.06265, 'acc_pick': 0.5242, 'acc_market': 0.5242, 'acc_model': 0.516},
            'confirm': {'seasons': [2019, 2025], 'matches': 1361, 'pick': 1.00263, 'pick_recalibrated': 1.00423, 'market': 1.00263, 'model': 1.02021, 'base': 1.0919, 'acc_pick': 0.5004, 'acc_market': 0.5004, 'acc_model': 0.496},
            'scores': {
                'dev': {
                    'market': {},
                    'model': {'btts': {'log_loss': 0.6903, 'base': 0.69216}, 'o15': {'log_loss': 0.51963, 'base': 0.52759}, 'o25': {'log_loss': 0.6825, 'base': 0.69025}, 'o35': {'log_loss': 0.6285, 'base': 0.63844}, 'exact': {'log_loss': 2.97407, 'base': 3.09975}},
                },
                'confirm': {
                    'market': {},
                    'model': {'btts': {'log_loss': 0.68591, 'base': 0.68865}, 'o15': {'log_loss': 0.50647, 'base': 0.50884}, 'o25': {'log_loss': 0.68381, 'base': 0.68817}, 'o35': {'log_loss': 0.62831, 'base': 0.63758}, 'exact': {'log_loss': 2.96855, 'base': 3.07066}},
                },
            },
        },
    },
    'BRA': {
        'name': 'Série A',
        'country': 'Brazil',
        'group': 'extra',
        'in_app': True,
        'pick': 'recalibrated',
        'model_alone': True,
        'numbers': {'market': None, 'model': ['o15', 'o25', 'o35', 'exact']},
        'weights': {'season': 2026, 'w': [1.1064, 0.0298, -0.0009]},
        'caveats': [
            "Measured on closing prices, the only ones its history has; its live pick uses the pre-match price. On the European divisions' 2019-2025, where both exist, the pre-match price scored 0.0035 worse in log loss than the closing one (1.0026 against 0.9990, 53,270 matches; accuracy 50.4% against 50.7%): its live picks should do about that much worse than measured, and its band rates were measured on closing-price picks.",
            'No shots in its history: the model rates its clubs on goals alone.',
            "No over/under price: every goals and scores number is the model's own.",
            "Not shown from the model alone: both teams to score - it did not beat the league's base rate in both 2013-2018 and 2019-2025.",
        ],
        'measured': {
            'price': 'closing',
            'season_check': {'last_result': '2026-10-30', 'days_since': 3, 'fixtures_posted': 3},
            'dev': {'seasons': [2013, 2018], 'matches': 2279, 'pick': 0.9933, 'pick_recalibrated': 0.9933, 'market': 0.99459, 'model': 1.01189, 'base': 1.03611, 'acc_pick': 0.5208, 'acc_market': 0.5204, 'acc_model': 0.5151},
            'confirm': {'seasons': [2019, 2025], 'matches': 2660, 'pick': 1.00008, 'pick_recalibrated': 1.00008, 'market': 1.00068, 'model': 1.0191, 'base': 1.06101, 'acc_pick': 0.5102, 'acc_market': 0.5109, 'acc_model': 0.488},
            'scores': {
                'dev': {
                    'market': {},
                    'model': {'btts': {'log_loss': 0.68616, 'base': 0.69128}, 'o15': {'log_loss': 0.63047, 'base': 0.6355}, 'o25': {'log_loss': 0.67494, 'base': 0.68172}, 'o35': {'log_loss': 0.50759, 'base': 0.51182}, 'exact': {'log_loss': 2.71387, 'base': 2.76237}},
                },
                'confirm': {
                    'market': {},
                    'model': {'btts': {'log_loss': 0.69762, 'base': 0.69375}, 'o15': {'log_loss': 0.60629, 'base': 0.61304}, 'o25': {'log_loss': 0.68136, 'base': 0.68637}, 'o35': {'log_loss': 0.50295, 'base': 0.50785}, 'exact': {'log_loss': 2.72686, 'base': 2.78226}},
                },
            },
        },
    },
    'CHN': {
        'name': 'Super League',
        'country': 'China',
        'group': 'extra',
        'in_app': True,
        'pick': 'recalibrated',
        'model_alone': True,
        'numbers': {'market': None, 'model': ['btts', 'o15', 'o25', 'o35', 'exact']},
        'weights': {'season': 2026, 'w': [1.1071, 0.0293, -0.003]},
        'caveats': [
            "Measured on closing prices, the only ones its history has; its live pick uses the pre-match price. On the European divisions' 2019-2025, where both exist, the pre-match price scored 0.0035 worse in log loss than the closing one (1.0026 against 0.9990, 53,270 matches; accuracy 50.4% against 50.7%): its live picks should do about that much worse than measured, and its band rates were measured on closing-price picks.",
            'No shots in its history: the model rates its clubs on goals alone.',
            "No over/under price: every goals and scores number is the model's own.",
            'Its history starts in 2014: that season only warms the base rates up.',
        ],
        'measured': {
            'price': 'closing',
            'season_check': {'last_result': '2026-10-24', 'days_since': 9, 'fixtures_posted': 0},
            'dev': {'seasons': [2013, 2018], 'matches': 960, 'pick': 0.96759, 'pick_recalibrated': 0.96759, 'market': 0.96873, 'model': 1.00236, 'base': 1.06083, 'acc_pick': 0.5271, 'acc_market': 0.5271, 'acc_model': 0.5073},
            'confirm': {'seasons': [2019, 2025], 'matches': 1602, 'pick': 0.89773, 'pick_recalibrated': 0.89773, 'market': 0.90209, 'model': 0.95422, 'base': 1.06851, 'acc_pick': 0.5787, 'acc_market': 0.5787, 'acc_model': 0.5537},
            'scores': {
                'dev': {
                    'market': {},
                    'model': {'btts': {'log_loss': 0.67823, 'base': 0.68017}, 'o15': {'log_loss': 0.497, 'base': 0.51028}, 'o25': {'log_loss': 0.67539, 'base': 0.68778}, 'o35': {'log_loss': 0.61743, 'base': 0.62747}, 'exact': {'log_loss': 2.94579, 'base': 3.04896}},
                },
                'confirm': {
                    'market': {},
                    'model': {'btts': {'log_loss': 0.68383, 'base': 0.6869}, 'o15': {'log_loss': 0.51495, 'base': 0.52085}, 'o25': {'log_loss': 0.67139, 'base': 0.68575}, 'o35': {'log_loss': 0.62336, 'base': 0.63939}, 'exact': {'log_loss': 2.90798, 'base': 3.07053}},
                },
            },
        },
    },
    'DNK': {
        'name': 'Superliga',
        'country': 'Denmark',
        'group': 'extra',
        'in_app': True,
        'pick': 'recalibrated',
        'model_alone': True,
        'numbers': {'market': None, 'model': ['btts', 'o15', 'o25', 'exact']},
        'weights': {'season': 2026, 'w': [1.1039, 0.0289, -0.0044]},
        'caveats': [
            "Measured on closing prices, the only ones its history has; its live pick uses the pre-match price. On the European divisions' 2019-2025, where both exist, the pre-match price scored 0.0035 worse in log loss than the closing one (1.0026 against 0.9990, 53,270 matches; accuracy 50.4% against 50.7%): its live picks should do about that much worse than measured, and its band rates were measured on closing-price picks.",
            'No shots in its history: the model rates its clubs on goals alone.',
            "No over/under price: every goals and scores number is the model's own.",
            "Not shown from the model alone: over 3.5 - it did not beat the league's base rate in both 2013-2018 and 2019-2025.",
        ],
        'measured': {
            'price': 'closing',
            'season_check': {'last_result': '2026-10-26', 'days_since': 7, 'fixtures_posted': 6},
            'dev': {'seasons': [2013, 2018], 'matches': 1355, 'pick': 1.00516, 'pick_recalibrated': 1.00516, 'market': 1.0062, 'model': 1.02458, 'base': 1.07559, 'acc_pick': 0.4967, 'acc_market': 0.4989, 'acc_model': 0.4878},
            'confirm': {'seasons': [2019, 2025], 'matches': 1399, 'pick': 1.01032, 'pick_recalibrated': 1.01032, 'market': 1.01109, 'model': 1.03004, 'base': 1.07803, 'acc_pick': 0.5139, 'acc_market': 0.5139, 'acc_model': 0.4918},
            'scores': {
                'dev': {
                    'market': {},
                    'model': {'btts': {'log_loss': 0.68788, 'base': 0.69265}, 'o15': {'log_loss': 0.55709, 'base': 0.55993}, 'o25': {'log_loss': 0.68723, 'base': 0.69458}, 'o35': {'log_loss': 0.60659, 'base': 0.61087}, 'exact': {'log_loss': 2.90849, 'base': 2.99832}},
                },
                'confirm': {
                    'market': {},
                    'model': {'btts': {'log_loss': 0.68058, 'base': 0.68162}, 'o15': {'log_loss': 0.50454, 'base': 0.506}, 'o25': {'log_loss': 0.68762, 'base': 0.68874}, 'o35': {'log_loss': 0.63094, 'base': 0.63084}, 'exact': {'log_loss': 2.92942, 'base': 2.99356}},
                },
            },
        },
    },
    'FIN': {
        'name': 'Veikkausliiga',
        'country': 'Finland',
        'group': 'extra',
        'in_app': True,
        'pick': 'raw',
        'model_alone': True,
        'numbers': {'market': None, 'model': ['o15', 'o25', 'o35', 'exact']},
        'weights': None,
        'caveats': [
            "Measured on closing prices, the only ones its history has; its live pick uses the pre-match price. On the European divisions' 2019-2025, where both exist, the pre-match price scored 0.0035 worse in log loss than the closing one (1.0026 against 0.9990, 53,270 matches; accuracy 50.4% against 50.7%): its live picks should do about that much worse than measured, and its band rates were measured on closing-price picks.",
            'No shots in its history: the model rates its clubs on goals alone.',
            "No over/under price: every goals and scores number is the model's own.",
            'The recalibrated market lost to the raw market on 2013-2018 (1.0158 against 1.0156), so its pick is the raw no-vig market.',
            "Not shown from the model alone: both teams to score - it did not beat the league's base rate in both 2013-2018 and 2019-2025.",
        ],
        'measured': {
            'price': 'closing',
            'season_check': {'last_result': '2026-10-29', 'days_since': 4, 'fixtures_posted': 3},
            'dev': {'seasons': [2013, 2018], 'matches': 1196, 'pick': 1.0156, 'pick_recalibrated': 1.01581, 'market': 1.0156, 'model': 1.03237, 'base': 1.07808, 'acc_pick': 0.4875, 'acc_market': 0.4875, 'acc_model': 0.4699},
            'confirm': {'seasons': [2019, 2025], 'matches': 1150, 'pick': 0.98514, 'pick_recalibrated': 0.98328, 'market': 0.98514, 'model': 1.01099, 'base': 1.0787, 'acc_pick': 0.54, 'acc_market': 0.54, 'acc_model': 0.5148},
            'scores': {
                'dev': {
                    'market': {},
                    'model': {'btts': {'log_loss': 0.69291, 'base': 0.69284}, 'o15': {'log_loss': 0.60434, 'base': 0.60514}, 'o25': {'log_loss': 0.6875, 'base': 0.69035}, 'o35': {'log_loss': 0.56116, 'base': 0.56396}, 'exact': {'log_loss': 2.84219, 'base': 2.92703}},
                },
                'confirm': {
                    'market': {},
                    'model': {'btts': {'log_loss': 0.68768, 'base': 0.69445}, 'o15': {'log_loss': 0.54156, 'base': 0.54856}, 'o25': {'log_loss': 0.68796, 'base': 0.69904}, 'o35': {'log_loss': 0.61377, 'base': 0.62237}, 'exact': {'log_loss': 2.91323, 'base': 3.00513}},
                },
            },
        },
    },
    'IRL': {
        'name': 'Premier Division',
        'country': 'Ireland',
        'group': 'extra',
        'in_app': True,
        'pick': 'recalibrated',
        'model_alone': True,
        'numbers': {'market': None, 'model': ['o15', 'o25', 'o35', 'exact']},
        'weights': {'season': 2026, 'w': [1.1063, 0.0299, -0.0011]},
        'caveats': [
            "Measured on closing prices, the only ones its history has; its live pick uses the pre-match price. On the European divisions' 2019-2025, where both exist, the pre-match price scored 0.0035 worse in log loss than the closing one (1.0026 against 0.9990, 53,270 matches; accuracy 50.4% against 50.7%): its live picks should do about that much worse than measured, and its band rates were measured on closing-price picks.",
            'No shots in its history: the model rates its clubs on goals alone.',
            "No over/under price: every goals and scores number is the model's own.",
            'On 2019-2025 the recalibrated pick scored a little worse than the raw market (0.9836 against 0.9832).',
            "Not shown from the model alone: both teams to score - it did not beat the league's base rate in both 2013-2018 and 2019-2025.",
        ],
        'measured': {
            'price': 'closing',
            'season_check': {'last_result': '2026-10-23', 'days_since': 10, 'fixtures_posted': 5},
            'dev': {'seasons': [2013, 2018], 'matches': 1191, 'pick': 0.9329, 'pick_recalibrated': 0.9329, 'market': 0.93458, 'model': 0.95973, 'base': 1.06541, 'acc_pick': 0.5676, 'acc_market': 0.5684, 'acc_model': 0.5483},
            'confirm': {'seasons': [2019, 2025], 'matches': 1189, 'pick': 0.98355, 'pick_recalibrated': 0.98355, 'market': 0.98323, 'model': 1.0016, 'base': 1.07743, 'acc_pick': 0.5156, 'acc_market': 0.5156, 'acc_model': 0.5029},
            'scores': {
                'dev': {
                    'market': {},
                    'model': {'btts': {'log_loss': 0.69236, 'base': 0.69365}, 'o15': {'log_loss': 0.56709, 'base': 0.56881}, 'o25': {'log_loss': 0.69054, 'base': 0.6937}, 'o35': {'log_loss': 0.58615, 'base': 0.59436}, 'exact': {'log_loss': 2.90183, 'base': 3.03957}},
                },
                'confirm': {
                    'market': {},
                    'model': {'btts': {'log_loss': 0.69324, 'base': 0.69202}, 'o15': {'log_loss': 0.60741, 'base': 0.6138}, 'o25': {'log_loss': 0.68024, 'base': 0.69138}, 'o35': {'log_loss': 0.5206, 'base': 0.53179}, 'exact': {'log_loss': 2.72624, 'base': 2.84795}},
                },
            },
        },
    },
    'JPN': {
        'name': 'J1 League',
        'country': 'Japan',
        'group': 'extra',
        'in_app': True,
        'pick': 'recalibrated',
        'model_alone': True,
        'numbers': {'market': None, 'model': ['btts', 'o15', 'o25', 'o35', 'exact']},
        'weights': {'season': 2026, 'w': [1.1029, 0.0281, -0.0058]},
        'caveats': [
            "Measured on closing prices, the only ones its history has; its live pick uses the pre-match price. On the European divisions' 2019-2025, where both exist, the pre-match price scored 0.0035 worse in log loss than the closing one (1.0026 against 0.9990, 53,270 matches; accuracy 50.4% against 50.7%): its live picks should do about that much worse than measured, and its band rates were measured on closing-price picks.",
            'No shots in its history: the model rates its clubs on goals alone.',
            "No over/under price: every goals and scores number is the model's own.",
        ],
        'measured': {
            'price': 'closing',
            'season_check': {'last_result': '2026-10-25', 'days_since': 8, 'fixtures_posted': 10},
            'dev': {'seasons': [2013, 2018], 'matches': 1845, 'pick': 1.02857, 'pick_recalibrated': 1.02857, 'market': 1.02859, 'model': 1.03934, 'base': 1.07473, 'acc_pick': 0.4959, 'acc_market': 0.4959, 'acc_model': 0.4726},
            'confirm': {'seasons': [2019, 2025], 'matches': 2372, 'pick': 1.0338, 'pick_recalibrated': 1.0338, 'market': 1.03441, 'model': 1.04749, 'base': 1.08279, 'acc_pick': 0.4718, 'acc_market': 0.4701, 'acc_model': 0.4688},
            'scores': {
                'dev': {
                    'market': {},
                    'model': {'btts': {'log_loss': 0.68656, 'base': 0.69523}, 'o15': {'log_loss': 0.59344, 'base': 0.59425}, 'o25': {'log_loss': 0.68901, 'base': 0.6939}, 'o35': {'log_loss': 0.58119, 'base': 0.586}, 'exact': {'log_loss': 2.89475, 'base': 2.9619}},
                },
                'confirm': {
                    'market': {},
                    'model': {'btts': {'log_loss': 0.68864, 'base': 0.69388}, 'o15': {'log_loss': 0.59294, 'base': 0.60065}, 'o25': {'log_loss': 0.68431, 'base': 0.69268}, 'o35': {'log_loss': 0.55959, 'base': 0.56953}, 'exact': {'log_loss': 2.86346, 'base': 2.92207}},
                },
            },
        },
    },
    'MEX': {
        'name': 'Liga MX',
        'country': 'Mexico',
        'group': 'extra',
        'in_app': True,
        'pick': 'recalibrated',
        'model_alone': True,
        'numbers': {'market': None, 'model': ['o15', 'o25', 'o35', 'exact']},
        'weights': {'season': 2026, 'w': [1.1037, 0.0288, -0.0047]},
        'caveats': [
            "Measured on closing prices, the only ones its history has; its live pick uses the pre-match price. On the European divisions' 2019-2025, where both exist, the pre-match price scored 0.0035 worse in log loss than the closing one (1.0026 against 0.9990, 53,270 matches; accuracy 50.4% against 50.7%): its live picks should do about that much worse than measured, and its band rates were measured on closing-price picks.",
            'No shots in its history: the model rates its clubs on goals alone.',
            "No over/under price: every goals and scores number is the model's own.",
            "Not shown from the model alone: both teams to score - it did not beat the league's base rate in both 2013-2018 and 2019-2025.",
        ],
        'measured': {
            'price': 'closing',
            'season_check': {'last_result': '2026-10-29', 'days_since': 4, 'fixtures_posted': 9},
            'dev': {'seasons': [2013, 2018], 'matches': 2004, 'pick': 1.0365, 'pick_recalibrated': 1.0365, 'market': 1.03694, 'model': 1.04923, 'base': 1.07379, 'acc_pick': 0.4716, 'acc_market': 0.4706, 'acc_model': 0.4521},
            'confirm': {'seasons': [2019, 2025], 'matches': 2317, 'pick': 1.00527, 'pick_recalibrated': 1.00527, 'market': 1.00667, 'model': 1.02236, 'base': 1.06829, 'acc_pick': 0.5037, 'acc_market': 0.5041, 'acc_model': 0.4985},
            'scores': {
                'dev': {
                    'market': {},
                    'model': {'btts': {'log_loss': 0.68933, 'base': 0.68939}, 'o15': {'log_loss': 0.58014, 'base': 0.58297}, 'o25': {'log_loss': 0.69176, 'base': 0.69462}, 'o35': {'log_loss': 0.589, 'base': 0.59613}, 'exact': {'log_loss': 2.85992, 'base': 2.9134}},
                },
                'confirm': {
                    'market': {},
                    'model': {'btts': {'log_loss': 0.68925, 'base': 0.68752}, 'o15': {'log_loss': 0.54898, 'base': 0.55397}, 'o25': {'log_loss': 0.68625, 'base': 0.69371}, 'o35': {'log_loss': 0.59779, 'base': 0.60285}, 'exact': {'log_loss': 2.87967, 'base': 2.94445}},
                },
            },
        },
    },
    'NOR': {
        'name': 'Eliteserien',
        'country': 'Norway',
        'group': 'extra',
        'in_app': True,
        'pick': 'recalibrated',
        'model_alone': True,
        'numbers': {'market': None, 'model': ['o15', 'o35', 'exact']},
        'weights': {'season': 2026, 'w': [1.1069, 0.0291, -0.0025]},
        'caveats': [
            "Measured on closing prices, the only ones its history has; its live pick uses the pre-match price. On the European divisions' 2019-2025, where both exist, the pre-match price scored 0.0035 worse in log loss than the closing one (1.0026 against 0.9990, 53,270 matches; accuracy 50.4% against 50.7%): its live picks should do about that much worse than measured, and its band rates were measured on closing-price picks.",
            'No shots in its history: the model rates its clubs on goals alone.',
            "No over/under price: every goals and scores number is the model's own.",
            "Not shown from the model alone: both teams to score, over 2.5 - it did not beat the league's base rate in both 2013-2018 and 2019-2025.",
        ],
        'measured': {
            'price': 'closing',
            'season_check': {'last_result': '2026-10-25', 'days_since': 8, 'fixtures_posted': 8},
            'dev': {'seasons': [2013, 2018], 'matches': 1451, 'pick': 0.99818, 'pick_recalibrated': 0.99818, 'market': 0.9986, 'model': 1.01222, 'base': 1.05512, 'acc_pick': 0.5217, 'acc_market': 0.5231, 'acc_model': 0.5176},
            'confirm': {'seasons': [2019, 2025], 'matches': 1689, 'pick': 0.97512, 'pick_recalibrated': 0.97512, 'market': 0.97608, 'model': 0.99418, 'base': 1.06036, 'acc_pick': 0.5263, 'acc_market': 0.5258, 'acc_model': 0.5163},
            'scores': {
                'dev': {
                    'market': {},
                    'model': {'btts': {'log_loss': 0.68117, 'base': 0.6808}, 'o15': {'log_loss': 0.50884, 'base': 0.51199}, 'o25': {'log_loss': 0.69089, 'base': 0.68836}, 'o35': {'log_loss': 0.63737, 'base': 0.63845}, 'exact': {'log_loss': 2.96157, 'base': 3.02555}},
                },
                'confirm': {
                    'market': {},
                    'model': {'btts': {'log_loss': 0.67704, 'base': 0.67983}, 'o15': {'log_loss': 0.48354, 'base': 0.49054}, 'o25': {'log_loss': 0.663, 'base': 0.67578}, 'o35': {'log_loss': 0.65322, 'base': 0.66665}, 'exact': {'log_loss': 3.0202, 'base': 3.12668}},
                },
            },
        },
    },
    'POL': {
        'name': 'Ekstraklasa',
        'country': 'Poland',
        'group': 'extra',
        'in_app': True,
        'pick': 'raw',
        'model_alone': True,
        'numbers': {'market': None, 'model': ['exact']},
        'weights': None,
        'caveats': [
            "Measured on closing prices, the only ones its history has; its live pick uses the pre-match price. On the European divisions' 2019-2025, where both exist, the pre-match price scored 0.0035 worse in log loss than the closing one (1.0026 against 0.9990, 53,270 matches; accuracy 50.4% against 50.7%): its live picks should do about that much worse than measured, and its band rates were measured on closing-price picks.",
            'No shots in its history: the model rates its clubs on goals alone.',
            "No over/under price: every goals and scores number is the model's own.",
            'The recalibrated market lost to the raw market on 2013-2018 (1.0455 against 1.0451), so its pick is the raw no-vig market.',
            "Not shown from the model alone: both teams to score, over 1.5, over 2.5, over 3.5 - it did not beat the league's base rate in both 2013-2018 and 2019-2025.",
        ],
        'measured': {
            'price': 'closing',
            'season_check': {'last_result': '2026-10-25', 'days_since': 8, 'fixtures_posted': 9},
            'dev': {'seasons': [2013, 2018], 'matches': 1776, 'pick': 1.04505, 'pick_recalibrated': 1.04553, 'market': 1.04505, 'model': 1.0584, 'base': 1.07575, 'acc_pick': 0.4634, 'acc_market': 0.4634, 'acc_model': 0.4583},
            'confirm': {'seasons': [2019, 2025], 'matches': 2066, 'pick': 1.03429, 'pick_recalibrated': 1.0344, 'market': 1.03429, 'model': 1.04943, 'base': 1.07329, 'acc_pick': 0.4656, 'acc_market': 0.4656, 'acc_model': 0.4593},
            'scores': {
                'dev': {
                    'market': {},
                    'model': {'btts': {'log_loss': 0.69296, 'base': 0.69172}, 'o15': {'log_loss': 0.55858, 'base': 0.55633}, 'o25': {'log_loss': 0.69592, 'base': 0.69388}, 'o35': {'log_loss': 0.60214, 'base': 0.60173}, 'exact': {'log_loss': 2.94128, 'base': 2.9793}},
                },
                'confirm': {
                    'market': {},
                    'model': {'btts': {'log_loss': 0.69062, 'base': 0.6915}, 'o15': {'log_loss': 0.57001, 'base': 0.57509}, 'o25': {'log_loss': 0.68599, 'base': 0.69314}, 'o35': {'log_loss': 0.58312, 'base': 0.58878}, 'exact': {'log_loss': 2.86964, 'base': 2.91464}},
                },
            },
        },
    },
    'ROU': {
        'name': 'SuperLiga',
        'country': 'Romania',
        'group': 'extra',
        'in_app': True,
        'pick': 'recalibrated',
        'model_alone': True,
        'numbers': {'market': None, 'model': ['btts', 'o15', 'o25', 'o35', 'exact']},
        'weights': {'season': 2026, 'w': [1.1037, 0.0288, -0.0047]},
        'caveats': [
            "Measured on closing prices, the only ones its history has; its live pick uses the pre-match price. On the European divisions' 2019-2025, where both exist, the pre-match price scored 0.0035 worse in log loss than the closing one (1.0026 against 0.9990, 53,270 matches; accuracy 50.4% against 50.7%): its live picks should do about that much worse than measured, and its band rates were measured on closing-price picks.",
            'No shots in its history: the model rates its clubs on goals alone.',
            "No over/under price: every goals and scores number is the model's own.",
        ],
        'measured': {
            'price': 'closing',
            'season_check': {'last_result': '2026-10-26', 'days_since': 7, 'fixtures_posted': 8},
            'dev': {'seasons': [2013, 2018], 'matches': 1694, 'pick': 0.99998, 'pick_recalibrated': 0.99998, 'market': 1.00079, 'model': 1.02767, 'base': 1.0821, 'acc_pick': 0.503, 'acc_market': 0.5018, 'acc_model': 0.49},
            'confirm': {'seasons': [2019, 2025], 'matches': 2181, 'pick': 1.00093, 'pick_recalibrated': 1.00093, 'market': 1.00212, 'model': 1.02767, 'base': 1.07958, 'acc_pick': 0.5002, 'acc_market': 0.4979, 'acc_model': 0.4718},
            'scores': {
                'dev': {
                    'market': {},
                    'model': {'btts': {'log_loss': 0.68959, 'base': 0.6956}, 'o15': {'log_loss': 0.62921, 'base': 0.63241}, 'o25': {'log_loss': 0.67972, 'base': 0.68477}, 'o35': {'log_loss': 0.50959, 'base': 0.51477}, 'exact': {'log_loss': 2.73235, 'base': 2.8282}},
                },
                'confirm': {
                    'market': {},
                    'model': {'btts': {'log_loss': 0.69106, 'base': 0.69357}, 'o15': {'log_loss': 0.61782, 'base': 0.62154}, 'o25': {'log_loss': 0.67645, 'base': 0.68456}, 'o35': {'log_loss': 0.53895, 'base': 0.5459}, 'exact': {'log_loss': 2.76402, 'base': 2.83218}},
                },
            },
        },
    },
    'RUS': {
        'name': 'Premier League',
        'country': 'Russia',
        'group': 'extra',
        'in_app': False,
        'pick': 'recalibrated',
        'model_alone': True,
        'numbers': {'market': None, 'model': ['btts', 'o15', 'o25', 'o35', 'exact']},
        'weights': {'season': 2026, 'w': [1.1039, 0.0289, -0.0044]},
        'caveats': [
            'Not in the app: no current season in the data - its newest result is 2026-08-02 (92 days before the check of 2026-11-02) and no fixture is posted. It is measured, and comes in by the same rule when its file resumes.',
            "Measured on closing prices, the only ones its history has; its live pick uses the pre-match price. On the European divisions' 2019-2025, where both exist, the pre-match price scored 0.0035 worse in log loss than the closing one (1.0026 against 0.9990, 53,270 matches; accuracy 50.4% against 50.7%): its live picks should do about that much worse than measured, and its band rates were measured on closing-price picks.",
            'No shots in its history: the model rates its clubs on goals alone.',
            "No over/under price: every goals and scores number is the model's own.",
        ],
        'measured': {
            'price': 'closing',
            'season_check': {'last_result': '2026-08-02', 'days_since': 92, 'fixtures_posted': 0},
            'dev': {'seasons': [2013, 2018], 'matches': 1464, 'pick': 0.98153, 'pick_recalibrated': 0.98153, 'market': 0.98201, 'model': 1.0032, 'base': 1.07737, 'acc_pick': 0.5362, 'acc_market': 0.5335, 'acc_model': 0.5109},
            'confirm': {'seasons': [2019, 2025], 'matches': 1610, 'pick': 0.98479, 'pick_recalibrated': 0.98479, 'market': 0.98566, 'model': 0.99919, 'base': 1.0715, 'acc_pick': 0.5205, 'acc_market': 0.5199, 'acc_model': 0.5261},
            'scores': {
                'dev': {
                    'market': {},
                    'model': {'btts': {'log_loss': 0.68272, 'base': 0.68688}, 'o15': {'log_loss': 0.63716, 'base': 0.64769}, 'o25': {'log_loss': 0.67395, 'base': 0.68488}, 'o35': {'log_loss': 0.51684, 'base': 0.52351}, 'exact': {'log_loss': 2.71486, 'base': 2.83746}},
                },
                'confirm': {
                    'market': {},
                    'model': {'btts': {'log_loss': 0.68897, 'base': 0.69972}, 'o15': {'log_loss': 0.55826, 'base': 0.57301}, 'o25': {'log_loss': 0.68384, 'base': 0.69805}, 'o35': {'log_loss': 0.58075, 'base': 0.59584}, 'exact': {'log_loss': 2.84126, 'base': 2.95423}},
                },
            },
        },
    },
    'SWE': {
        'name': 'Allsvenskan',
        'country': 'Sweden',
        'group': 'extra',
        'in_app': True,
        'pick': 'recalibrated',
        'model_alone': True,
        'numbers': {'market': None, 'model': ['btts', 'o15', 'o25', 'o35', 'exact']},
        'weights': {'season': 2026, 'w': [1.1054, 0.027, -0.0046]},
        'caveats': [
            "Measured on closing prices, the only ones its history has; its live pick uses the pre-match price. On the European divisions' 2019-2025, where both exist, the pre-match price scored 0.0035 worse in log loss than the closing one (1.0026 against 0.9990, 53,270 matches; accuracy 50.4% against 50.7%): its live picks should do about that much worse than measured, and its band rates were measured on closing-price picks.",
            'No shots in its history: the model rates its clubs on goals alone.',
            "No over/under price: every goals and scores number is the model's own.",
        ],
        'measured': {
            'price': 'closing',
            'season_check': {'last_result': '2026-10-29', 'days_since': 4, 'fixtures_posted': 8},
            'dev': {'seasons': [2013, 2018], 'matches': 1452, 'pick': 0.97593, 'pick_recalibrated': 0.97593, 'market': 0.97703, 'model': 1.004, 'base': 1.07211, 'acc_pick': 0.5296, 'acc_market': 0.5296, 'acc_model': 0.5055},
            'confirm': {'seasons': [2019, 2025], 'matches': 1690, 'pick': 0.98381, 'pick_recalibrated': 0.98381, 'market': 0.98464, 'model': 1.00858, 'base': 1.07207, 'acc_pick': 0.5189, 'acc_market': 0.5189, 'acc_model': 0.4994},
            'scores': {
                'dev': {
                    'market': {},
                    'model': {'btts': {'log_loss': 0.68432, 'base': 0.68542}, 'o15': {'log_loss': 0.51791, 'base': 0.52002}, 'o25': {'log_loss': 0.68493, 'base': 0.68821}, 'o35': {'log_loss': 0.62168, 'base': 0.62671}, 'exact': {'log_loss': 2.91905, 'base': 3.01891}},
                },
                'confirm': {
                    'market': {},
                    'model': {'btts': {'log_loss': 0.68807, 'base': 0.69325}, 'o15': {'log_loss': 0.53174, 'base': 0.54061}, 'o25': {'log_loss': 0.68094, 'base': 0.69228}, 'o35': {'log_loss': 0.59865, 'base': 0.60725}, 'exact': {'log_loss': 2.90785, 'base': 3.00376}},
                },
            },
        },
    },
    'SWZ': {
        'name': 'Super League',
        'country': 'Switzerland',
        'group': 'extra',
        'in_app': True,
        'pick': 'recalibrated',
        'model_alone': True,
        'numbers': {'market': None, 'model': ['btts', 'o25', 'o35', 'exact']},
        'weights': {'season': 2026, 'w': [1.104, 0.0287, -0.0046]},
        'caveats': [
            "Measured on closing prices, the only ones its history has; its live pick uses the pre-match price. On the European divisions' 2019-2025, where both exist, the pre-match price scored 0.0035 worse in log loss than the closing one (1.0026 against 0.9990, 53,270 matches; accuracy 50.4% against 50.7%): its live picks should do about that much worse than measured, and its band rates were measured on closing-price picks.",
            'No shots in its history: the model rates its clubs on goals alone.',
            "No over/under price: every goals and scores number is the model's own.",
            'On 2019-2025 the recalibrated pick scored a little worse than the raw market (1.0260 against 1.0256).',
            "Not shown from the model alone: over 1.5 - it did not beat the league's base rate in both 2013-2018 and 2019-2025.",
        ],
        'measured': {
            'price': 'closing',
            'season_check': {'last_result': '2026-10-25', 'days_since': 8, 'fixtures_posted': 6},
            'dev': {'seasons': [2013, 2018], 'matches': 1082, 'pick': 0.99572, 'pick_recalibrated': 0.99572, 'market': 0.99687, 'model': 1.0125, 'base': 1.07355, 'acc_pick': 0.5259, 'acc_market': 0.5259, 'acc_model': 0.5055},
            'confirm': {'seasons': [2019, 2025], 'matches': 1412, 'pick': 1.02598, 'pick_recalibrated': 1.02598, 'market': 1.02556, 'model': 1.04606, 'base': 1.07714, 'acc_pick': 0.4894, 'acc_market': 0.4887, 'acc_model': 0.4632},
            'scores': {
                'dev': {
                    'market': {},
                    'model': {'btts': {'log_loss': 0.67828, 'base': 0.68525}, 'o15': {'log_loss': 0.49435, 'base': 0.4986}, 'o25': {'log_loss': 0.67322, 'base': 0.68655}, 'o35': {'log_loss': 0.65849, 'base': 0.66541}, 'exact': {'log_loss': 3.02572, 'base': 3.12798}},
                },
                'confirm': {
                    'market': {},
                    'model': {'btts': {'log_loss': 0.66936, 'base': 0.66941}, 'o15': {'log_loss': 0.47322, 'base': 0.47263}, 'o25': {'log_loss': 0.6765, 'base': 0.67901}, 'o35': {'log_loss': 0.65583, 'base': 0.65685}, 'exact': {'log_loss': 3.00647, 'base': 3.06101}},
                },
            },
        },
    },
    'USA': {
        'name': 'MLS',
        'country': 'USA',
        'group': 'extra',
        'in_app': True,
        'pick': 'recalibrated',
        'model_alone': True,
        'numbers': {'market': None, 'model': ['btts', 'o15', 'o25', 'o35', 'exact']},
        'weights': {'season': 2026, 'w': [1.1064, 0.0301, -0.0019]},
        'caveats': [
            "Measured on closing prices, the only ones its history has; its live pick uses the pre-match price. On the European divisions' 2019-2025, where both exist, the pre-match price scored 0.0035 worse in log loss than the closing one (1.0026 against 0.9990, 53,270 matches; accuracy 50.4% against 50.7%): its live picks should do about that much worse than measured, and its band rates were measured on closing-price picks.",
            'No shots in its history: the model rates its clubs on goals alone.',
            "No over/under price: every goals and scores number is the model's own.",
            'On 2019-2025 the recalibrated pick scored a little worse than the raw market (1.0230 against 1.0226).',
        ],
        'measured': {
            'price': 'closing',
            'season_check': {'last_result': '2026-10-29', 'days_since': 4, 'fixtures_posted': 15},
            'dev': {'seasons': [2013, 2018], 'matches': 2189, 'pick': 0.99494, 'pick_recalibrated': 0.99494, 'market': 0.99681, 'model': 1.01081, 'base': 1.02281, 'acc_pick': 0.5254, 'acc_market': 0.5231, 'acc_model': 0.5231},
            'confirm': {'seasons': [2019, 2025], 'matches': 3288, 'pick': 1.02299, 'pick_recalibrated': 1.02299, 'market': 1.02261, 'model': 1.04045, 'base': 1.05859, 'acc_pick': 0.4918, 'acc_market': 0.4921, 'acc_model': 0.4872},
            'scores': {
                'dev': {
                    'market': {},
                    'model': {'btts': {'log_loss': 0.68536, 'base': 0.68677}, 'o15': {'log_loss': 0.524, 'base': 0.52932}, 'o25': {'log_loss': 0.68499, 'base': 0.68869}, 'o35': {'log_loss': 0.62575, 'base': 0.63296}, 'exact': {'log_loss': 2.95068, 'base': 2.98195}},
                },
                'confirm': {
                    'market': {},
                    'model': {'btts': {'log_loss': 0.67652, 'base': 0.67731}, 'o15': {'log_loss': 0.51418, 'base': 0.51739}, 'o25': {'log_loss': 0.67962, 'base': 0.6822}, 'o35': {'log_loss': 0.63647, 'base': 0.63926}, 'exact': {'log_loss': 2.99015, 'base': 3.01644}},
                },
            },
        },
    },
}
# Rule (d): each group's band hit rates on its confirmation period (the leagues in the app):
# the pick's ("pick") and the model's own picks' at the same edges ("model").
BANDS = {
    'top': {
        'pick': {'STRONG': {'hit_rate': 0.7116, 'picks': 5648}, 'SOLID': {'hit_rate': 0.5463, 'picks': 3908}, 'LEAN': {'hit_rate': 0.4508, 'picks': 5444}, 'TOSS-UP': {'hit_rate': 0.3723, 'picks': 2936}},
        'model': {'STRONG': {'hit_rate': 0.7142, 'picks': 4710}, 'SOLID': {'hit_rate': 0.5533, 'picks': 4379}, 'LEAN': {'hit_rate': 0.4564, 'picks': 5958}, 'TOSS-UP': {'hit_rate': 0.3676, 'picks': 2889}},
    },
    'europe': {
        'pick': {'STRONG': {'hit_rate': 0.7069, 'picks': 9511}, 'SOLID': {'hit_rate': 0.5476, 'picks': 12446}, 'LEAN': {'hit_rate': 0.4467, 'picks': 23231}, 'TOSS-UP': {'hit_rate': 0.3704, 'picks': 13434}},
        'model': {'STRONG': {'hit_rate': 0.7369, 'picks': 5799}, 'SOLID': {'hit_rate': 0.5657, 'picks': 11515}, 'LEAN': {'hit_rate': 0.4506, 'picks': 27556}, 'TOSS-UP': {'hit_rate': 0.3701, 'picks': 13752}},
    },
    'extra': {
        'pick': {'STRONG': {'hit_rate': 0.6894, 'picks': 5915}, 'SOLID': {'hit_rate': 0.5388, 'picks': 7066}, 'LEAN': {'hit_rate': 0.4437, 'picks': 10814}, 'TOSS-UP': {'hit_rate': 0.3589, 'picks': 5746}},
        'model': {'STRONG': {'hit_rate': 0.6714, 'picks': 4221}, 'SOLID': {'hit_rate': 0.535, 'picks': 7376}, 'LEAN': {'hit_rate': 0.4435, 'picks': 12190}, 'TOSS-UP': {'hit_rate': 0.3704, 'picks': 5754}},
    },
}
# Pre-match against closing no-vig log loss, the European divisions 2019-2025 (same matches).
PRICE_GAP = {'all': {'matches': 53270, 'pre': 1.00256, 'close': 0.99905, 'gap': 0.00351, 'gap_se': 0.00031, 'acc_pre': 0.5036, 'acc_close': 0.5067}, 'top': {'matches': 12458, 'pre': 0.97294, 'close': 0.97051, 'gap': 0.00243, 'gap_se': 0.00059, 'acc_pre': 0.5359, 'acc_close': 0.5396}, 'europe': {'matches': 40812, 'pre': 1.0116, 'close': 1.00776, 'gap': 0.00385, 'gap_se': 0.00036, 'acc_pre': 0.4938, 'acc_close': 0.4967}}


# ----------------------------------------------------------- reading them

def in_app(matches=None, fixtures=None, today=None):
    """The leagues the app carries, in soccer_data's order: as decided on
    CHECKED_ON - or, given the data and the day, every measured league whose
    current season rule (a) finds in that data now (a league's season ends,
    RUS's file resumes: its measured numbers and decisions stand)."""
    if matches is None:
        return [c for c in sd.ALL_LEAGUES if c in SHIPPED and SHIPPED[c]["in_app"]]
    return [c for c in sd.ALL_LEAGUES if c in SHIPPED
            and season_check(matches, fixtures, c, today or CHECKED_ON)["ok"]]


def raw_leagues(leagues=None):
    """Of `leagues` (default in_app()), those whose pick is the raw no-vig
    market (rule a)."""
    return [c for c in (in_app() if leagues is None else leagues) if SHIPPED[c]["pick"] == "raw"]


def model_alone(leagues=None):
    """Of `leagues` (default in_app()), those whose unpriced matches get the
    model's own pick (rule b)."""
    return [c for c in (in_app() if leagues is None else leagues) if SHIPPED[c]["model_alone"]]


def withheld(code, source):
    """The NUMBERS a card of `code` whose score matrix comes from `source`
    ("market" / "model") must not show (rule c) - every one for a league or
    a source that was not measured."""
    shown = (SHIPPED.get(code) or {}).get("numbers", {}).get(source)
    return tuple(k for k in NUMBERS if not shown or k not in shown)


def band_rate(code, band, basis="pick"):
    """The measured hit rate (and count) of `band` for `code`'s group on the
    confirmation period: basis "pick" for a market pick, "model" for a pick
    by the model alone. None where not measured."""
    g = BANDS.get(group_of(code)) or {}
    return (g.get(basis) or {}).get(band)


def caveats(code):
    """The caveats a league's cards and the Method tab carry (never deleted)."""
    return list((SHIPPED.get(code) or {}).get("caveats", []))
