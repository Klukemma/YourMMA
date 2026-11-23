"""Same-game tickets: four parlays per NFL game, built before kickoff from the
engine's own chances, with every leg from that one game.

PRE-REGISTERED 2026-11-23, before any ticket was built or any result
computed. The ticket shapes below are the user's (small, medium, large,
rookie); the rule fixes how each slot is filled, so that the backtest
replays exactly what the app shows. Changing a number here after the
backtest has run is a new version (V2...), with this one kept and measured
beside it.

THE LEGS (v1). Four kinds, each with a chance the engine has measured and a
price the market posts:

  WIN     a team to win (moneyline, overtime included). Chance: the pick's
          blend - P(home) = the walk-forward blend_prob, away = 1 - it
          (nfl_model; 66.4% of winners and log loss 0.6080 on 2016-2025,
          the closing favourite 66.6% and 0.6085). A tie voids the leg.
  SPREAD  a team at the posted spread (the handicap). Chance: the market's
          no-vig at the consensus spread prices - the model does not beat
          the line (its own cover chance scored 0.707 log loss on
          2016-2025, worse than a coin's 0.693). A push voids.
  TOTAL   over or under the posted game total. Chance: the market's no-vig
          at the consensus total prices. A push voids.
  TD      a player scores 1+ rushing or receiving touchdown, overtime
          included - passing, return and fumble-recovery touchdowns do not
          count, which is also Stake's "anytime touchdown" rule. Chance:
          nfl_td's walk-forward probability, which is conditional on the
          player playing; only players in a usage group the walk-forward
          measured as calibrated (nfl_td.shipped_groups: a live chance that
          is not None). A player who does not play voids the leg, as Stake
          voids a player listed inactive.

  A ROOKIE leg is a TD leg on a player in his first NFL season: no game in
  player_games in any earlier season. Live and in the backtest alike.

  Not in v1, because nothing on file measures them yet: QB pass attempts,
  yards over a line, first/second-half points, team or game touchdown
  totals, alternate ("Asian") lines. Each needs a probability measured
  against outcomes and a posted price; they arrive as V2 legs with their
  own measurement, never as an unmeasured guess.

THE TICKETS. Built in this order, each slot taking the highest-chance leg
still AVAILABLE (ties: the leg key, alphabetically). A leg is available to
a ticket while it is in fewer than USE_CAP of the game's tickets, so one
bad leg sinks at most two of the four. Within a ticket: one result leg
(WIN or SPREAD, either team), one TOTAL leg, and TD legs on different
players.

  SMALL   (2 legs) the best TD + the best result leg.
  ROOKIE  (2 legs) the best rookie TD with a chance of ROOKIE_FLOOR or more
          + WIN for the rookie's own team. A ticket identical to SMALL's
          moves to the next qualifying rookie. No qualifying rookie, or
          that WIN no longer available: no ticket, and the reason.
  MEDIUM  (3 legs) the best TD + the best result leg + the best of the
          TOTAL legs and the TD legs on other players.
  LARGE   (4 legs) the best TD + the next-best TD (another player) + the
          best result leg + the best TOTAL leg.

  A slot with nothing available leaves its ticket unbuilt, with the reason
  shown; a ticket is never filled with a leg outside its rule.

THE CHANCE AND THE PRICE.
  chance        the product of the legs' chances - what the ticket would
                hit if its legs were independent. Legs in one game are not
                (a team winning and its running back scoring go together),
                so the backtest measures how often each kind actually hit
                beside this product, and the app shows both.
  est. payout   the product of the legs' consensus single prices, marked
                "est.". Stake's bet builder prices correlated legs BELOW
                the product of the singles, so this is an upper bound on
                what Stake pays. The exact payout is the price Stake quotes
                for the slip: the app takes it as typed and computes the
                return on the stake from it.

THE ROI METERS. For each kind, independently, as if $10 had been bet on
every ticket of that kind:
  history   the backtest (experiments/nfl_sgp_backtest.py) over every game
            from the first season the TD walk-forward covers (2014) to the
            last one played, with the periods 2014-2019, 2020-2025 and the
            current season reported apart. Its inputs are point-in-time:
            the walk-forward blend and TD chances, the closing lines and
            prices (games.csv.gz), and the rookie flag. Tickets hit and
            tickets built are measured. Return needs every leg's price and
            no past anytime-TD price is on file, so return is shown as
            missing with that reason until a price history is added.
  live      every ticket logged before kickoff (prediction_history.json,
            the game's "tickets") and graded from the box score, its return
            at the logged est. payout (so an upper bound, and marked so).

  KNOWN BIAS, said where the meter is: in the backtest a TD candidate is a
  player who took a snap, so the replay knows who was active. Live, a pick
  who does not play is voided and the ticket is graded on its other legs.

Nothing here claims an edge. The market is hard to beat, and parlays
multiply the margin of every leg; these tickets are the legs the engine
rates most likely to land, not bets it rates profitable.
"""

VERSION = "sgp-v1"
KINDS = ("small", "rookie", "medium", "large")     # build order
LEGS = {"small": 2, "rookie": 2, "medium": 3, "large": 4}
USE_CAP = 2                    # a leg is in at most this many of a game's tickets
ROOKIE_FLOOR = 0.25            # a rookie TD chance this high or more qualifies
LEG_KINDS = ("WIN", "SPREAD", "TOTAL", "TD")
RESULT_KINDS = ("WIN", "SPREAD")
HISTORY_FROM = 2014            # the TD walk-forward's first scored season
PERIODS = {"2014-2019": (2014, 2019), "2020-2025": (2020, 2025)}
STAKE_UNIT = 10.0              # the meters' bet per ticket, in dollars
