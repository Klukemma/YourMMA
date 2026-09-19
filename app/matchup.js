// The matchup advantages, in the browser.
//
// A PORT, NOT A REIMPLEMENTATION. engine/matchup.py is the original and the
// only place these formulas are reasoned about; this file mirrors it so a
// phone can compare any two fighters without a server.
//
// Nothing here hardcodes a constant. Every number comes from data/constants.json,
// which the engine writes straight off matchup.py, because a hand-copied league
// baseline in a second language drifts silently and the app would go on
// computing a striking advantage against last year's numbers with nothing to
// say so. tests/test_app_js.py runs this file under node against advantages
// computed in Python and fails on the sixth decimal place.
//
// The NaN rule is the same one that governs the whole engine: a missing input
// produces a missing advantage, never a confident zero. An even matchup and an
// unmeasured one are different facts, and `known` is what separates them.

export function makeMatchup(C) {
  const finite = (x) => typeof x === "number" && Number.isFinite(x);

  // A fighter's rate as a multiple of the league's, bounded so that the
  // product of two extremes cannot assert a fight is impossible.
  function rateRatio(rate, league) {
    if (!finite(rate) || !finite(league) || league <= 0) return NaN;
    return Math.min(Math.max(rate / league, C.RATE_RATIO_FLOOR),
                    C.RATE_RATIO_CAP);
  }

  // Bill James' log5: the chance an attacker with rate `a` beats a defender
  // who concedes `b`, both expressed against the same league rate.
  function log5(attacker, defenderConceded, league) {
    if (!finite(attacker) || !finite(defenderConceded) || !finite(league)) {
      return NaN;
    }
    const eps = C.PROBABILITY_EPS;
    const a = Math.min(Math.max(attacker, eps), 1 - eps);
    const b = Math.min(Math.max(defenderConceded, eps), 1 - eps);
    const l = Math.min(Math.max(league, eps), 1 - eps);
    const numerator = a * b / l;
    const denominator = numerator + (1 - a) * (1 - b) / (1 - l);
    if (!finite(denominator) || denominator === 0) return NaN;
    return numerator / denominator;
  }

  function expectedSigAttempts(attacker, defender) {
    const ratio = rateRatio(attacker.sig_att_per_min, C.LEAGUE_SIG_ATT_PER_MIN);
    const faced = rateRatio(defender.sig_att_faced_per_min,
                            C.LEAGUE_SIG_ATT_PER_MIN);
    if (!finite(ratio) || !finite(faced)) return NaN;
    return C.LEAGUE_SIG_ATT_PER_MIN * ratio * faced;
  }

  function expectedSigAccuracy(attacker, defender) {
    return log5(attacker.sig_accuracy, defender.sig_accuracy_conceded,
                C.LEAGUE_SIG_ACC);
  }

  function expectedSigLanded(attacker, defender) {
    const attempts = expectedSigAttempts(attacker, defender);
    const accuracy = expectedSigAccuracy(attacker, defender);
    if (!finite(attempts) || !finite(accuracy)) return NaN;
    return attempts * accuracy;
  }

  function expectedTdAttempts(attacker, defender) {
    const ratio = rateRatio(attacker.td_att_per_min, C.LEAGUE_TD_ATT_PER_MIN);
    const faced = rateRatio(defender.td_att_faced_per_min,
                            C.LEAGUE_TD_ATT_PER_MIN);
    if (!finite(ratio) || !finite(faced)) return NaN;
    return C.LEAGUE_TD_ATT_PER_MIN * ratio * faced;
  }

  function expectedTakedowns(attacker, defender) {
    const attempts = expectedTdAttempts(attacker, defender);
    const accuracy = log5(attacker.td_accuracy, defender.td_accuracy_conceded,
                          C.LEAGUE_TD_ACC);
    if (!finite(attempts) || !finite(accuracy)) return NaN;
    return attempts * accuracy;
  }

  function expectedControl(attacker, defender) {
    const ratio = rateRatio(attacker.ctrl_sec_per_min,
                            C.LEAGUE_CTRL_SEC_PER_MIN);
    const conceded = rateRatio(defender.ctrl_sec_conceded_per_min,
                               C.LEAGUE_CTRL_SEC_PER_MIN);
    if (!finite(ratio) || !finite(conceded)) return NaN;
    return C.LEAGUE_CTRL_SEC_PER_MIN * ratio * conceded;
  }

  function expectedSubAttempts(attacker, defender) {
    const ratio = rateRatio(attacker.sub_att_per_min, C.LEAGUE_SUB_ATT_PER_MIN);
    const conceded = rateRatio(defender.sub_att_conceded_per_min,
                               C.LEAGUE_SUB_ATT_PER_MIN);
    if (!finite(ratio) || !finite(conceded)) return NaN;
    return C.LEAGUE_SUB_ATT_PER_MIN * ratio * conceded;
  }

  const difference = (f) => (red, blue) => {
    const mine = f(red, blue);
    const theirs = f(blue, red);
    if (!finite(mine) || !finite(theirs)) return NaN;
    return mine - theirs;
  };

  const strikingAdvantage = difference(expectedSigLanded);
  const takedownAdvantage = difference(expectedTakedowns);
  const controlAdvantage = difference(expectedControl);
  const submissionAdvantage = difference(expectedSubAttempts);

  function grapplingAdvantage(red, blue) {
    const ctrl = controlAdvantage(red, blue);
    const sub = submissionAdvantage(red, blue);
    const td = takedownAdvantage(red, blue);
    if (!finite(ctrl) || !finite(sub) || !finite(td)) return NaN;
    return C.GRAP_W_CTRL * (ctrl / C.CTRL_ADV_SD)
         + C.GRAP_W_SUB * (sub / C.SUB_ADV_SD)
         + C.GRAP_W_TD * (td / C.TD_ADV_SD);
  }

  // Never clipped. A 381 cm height is a scrape error, and clipping it to the
  // tallest fighter in UFC history would assert something instead of admitting
  // the value is unusable.
  const plausibleHeight = (cm) =>
    finite(cm) && cm >= C.HEIGHT_MIN_CM && cm <= C.HEIGHT_MAX_CM ? cm : NaN;
  const plausibleReach = (cm) =>
    finite(cm) && cm >= C.REACH_MIN_CM && cm <= C.REACH_MAX_CM ? cm : NaN;

  const heightAdvantage = (red, blue) => {
    const a = plausibleHeight(red.height_cm), b = plausibleHeight(blue.height_cm);
    return finite(a) && finite(b) ? a - b : NaN;
  };
  const reachAdvantage = (red, blue) => {
    const a = plausibleReach(red.reach_cm), b = plausibleReach(blue.reach_cm);
    return finite(a) && finite(b) ? a - b : NaN;
  };

  // Height and reach are weighted equally because three reasonable windows put
  // height's share at +0.21, -1.36 and 1.0. That is not an estimate.
  function sizeAdvantage(red, blue) {
    const h = heightAdvantage(red, blue);
    const r = reachAdvantage(red, blue);
    if (!finite(h) || !finite(r)) return NaN;
    return C.SIZE_W_HEIGHT * (h / C.HEIGHT_DIFF_SD_CM)
         + C.SIZE_W_REACH * (r / C.REACH_DIFF_SD_CM);
  }

  function massAdvantage(red, blue) {
    const a = red.weight_kg, b = blue.weight_kg;
    if (!finite(a) || !finite(b)) return NaN;
    return (a - b) / C.WEIGHT_DIFF_SD_KG;
  }

  return {
    rateRatio, log5,
    expectedSigLanded, expectedSigAccuracy, expectedSigAttempts,
    expectedTakedowns, expectedControl, expectedSubAttempts,
    strikingAdvantage, takedownAdvantage, controlAdvantage,
    submissionAdvantage, grapplingAdvantage,
    heightAdvantage, reachAdvantage, sizeAdvantage, massAdvantage,
  };
}

// The shrinkage that makes a thin record usable. Same weights as
// engine/matchup_inputs.py, and the same rule: no exposure means no rate, not
// a league-average one.
export function makeForm(C) {
  const finite = (x) => typeof x === "number" && Number.isFinite(x);

  // Takes the NUMERATOR, not the rate. Dividing by the exposure and
  // multiplying straight back is two extra floating-point operations that
  // engine/matchup_inputs.py does not perform, and the cross-language test
  // catches the difference at the ninth decimal place.
  function shrink(numerator, exposure, league, weight) {
    if (!finite(numerator) || !finite(exposure) || exposure <= 0) return NaN;
    return (numerator + league * weight) / (exposure + weight);
  }

  // [field, career column, league value, prior weight, unit scale]
  // A scale of null means the column is already a total rather than a rate.
  const RATES = [
    ["sig_att_per_min", "cd_sig_atmpted", "LEAGUE_SIG_ATT_PER_MIN",
     "K_STRIKE_VOLUME_MIN", null],
    ["sig_att_faced_per_min", "cd_opp_sig_atmpted", "LEAGUE_SIG_ATT_PER_MIN",
     "K_STRIKE_ABSORBED_MIN", null],
    ["td_att_per_min", "cd_td_atmpted", "LEAGUE_TD_ATT_PER_MIN",
     "K_TD_RATE_MIN", null],
    ["td_att_faced_per_min", "cd_opp_td_atmpted", "LEAGUE_TD_ATT_PER_MIN",
     "K_TD_CONCEDED_MIN", null],
    ["ctrl_sec_per_min", "cd_ctrl_share", "LEAGUE_CTRL_SEC_PER_MIN",
     "K_CTRL_MIN", 60],
    ["ctrl_sec_conceded_per_min", "cd_opp_ctrl_share",
     "LEAGUE_CTRL_SEC_PER_MIN", "K_CTRL_CONCEDED_MIN", 60],
    ["sub_att_per_min", "cd_sub_per15", "LEAGUE_SUB_ATT_PER_MIN",
     "K_SUB_ATT_MIN", 1 / 15],
    ["sub_att_conceded_per_min", "cd_opp_sub_per15", "LEAGUE_SUB_ATT_PER_MIN",
     "K_SUB_CONCEDED_MIN", 1 / 15],
  ];

  // [field, column, evidence column, league, weight, invert]
  const ACCURACIES = [
    ["sig_accuracy", "cd_str_acc", "cd_sig_atmpted", "LEAGUE_SIG_ACC",
     "K_SIG_ACC_ATT", false],
    ["sig_accuracy_conceded", "cd_str_def", "cd_opp_sig_atmpted",
     "LEAGUE_SIG_ACC", "K_SIG_DEF_ATT", true],
    ["td_accuracy", "cd_td_acc", "cd_td_atmpted", "LEAGUE_TD_ACC",
     "K_TD_ACC_ATT", false],
    ["td_accuracy_conceded", "cd_td_def", "cd_opp_td_atmpted",
     "LEAGUE_TD_ACC", "K_TD_DEF_ATT", true],
  ];

  return function form(stats) {
    const get = (k) => {
      const v = stats[k];
      return typeof v === "number" && Number.isFinite(v) ? v : NaN;
    };
    const minutes = get("cd_minutes");
    const out = {};
    for (const [field, column, league, weight, scale] of RATES) {
      const raw = get(column);
      // A published rate times its own exposure is the count that produced it.
      const numerator = scale === null ? raw : raw * scale * minutes;
      out[field] = shrink(numerator, minutes, C[league], C[weight]);
    }
    for (const [field, column, evidence, league, weight, invert] of ACCURACIES) {
      const attempts = get(evidence);
      let fraction = get(column);
      if (invert) fraction = 1 - fraction;
      out[field] = shrink(fraction * attempts, attempts, C[league], C[weight]);
    }
    out.height_cm = get("height_cm");
    out.reach_cm = get("reach_cm");
    out.weight_kg = get("weight_kg");
    return out;
  };
}
