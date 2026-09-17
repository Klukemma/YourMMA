"""
Debug helper: print a side-by-side stat comparison for two fighters.

Run this AFTER predict_card.py in the same interactive session (it relies on
fighter_stats / resolve_fighter_name built there):

    python -i engine/predict_card.py
    >>> exec(open("engine/compare_fighters.py").read())
"""

# ============================================================================
# DEBUG: FIGHTER COMPARISON
# ============================================================================
print("\n" + "="*70)
print("DEBUG: FIGHTER STATS COMPARISON")
print("="*70)

# Compare two fighters
fighter1 = "Justin Gaethje"
fighter2 = "Paddy Pimblett"

r = fighter_stats.get(_norm_name(fighter1))
b = fighter_stats.get(_norm_name(fighter2))

print(f"\n{fighter1} found: {r is not None}")
print(f"{fighter2} found: {b is not None}")

if r and b:
    print(f"\n{'Stat':<20} {fighter1:<20} {fighter2:<20} {'Diff':<15}")
    print("-" * 75)

    keys = ['mu', 'sigma', 'mmr_pre', 'wins', 'losses', 'opp_quality',
            'ko_rate', 'sub_rate', 'splm', 'str_acc', 'sapm', 'str_def',
            'td_avg', 'td_def', 'streak', 'momentum', 'won_L3', 'layoff']

    for key in keys:
        r_val = r.get(key, 0)
        b_val = b.get(key, 0)
        try:
            diff = float(r_val) - float(b_val)
            diff_str = f"{diff:+.2f}"
        except:
            diff_str = "N/A"
        print(f"{key:<20} {str(r_val):<20} {str(b_val):<20} {diff_str:<15}")
else:
    print("One or both fighters not found!")
    print(f"Available fighters sample: {list(fighter_stats.keys())[:10]}")
