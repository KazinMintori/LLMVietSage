"""Frozen research versions, feature schema and synthetic fixture settings."""

ALGORITHM = "C01-F1.0"

PROTOCOL = "C01-E1.0"

IMPLEMENTATION = "vietsage-proactive-1"

NEEDS = ("meal_coordination", "clothes_changing_drying", "rest_opportunity", "transport_coordination")

FEATURES = ("current_activity", "last_activity", "next_activity", "trip_phase", "local_time_block",
            "elapsed_time_bin", "free_time_bin", "coarse_location", "weather", "budget",
            "accessibility", "interruption_tolerance") + tuple(f"{n}_status" for n in NEEDS)

ANSWERS = ("yes", "no", "uncertain", "abort")

EPS = 1e-12

FIXTURE_PARAMETERS = dict(needs=4, horizon=60., history_window=1440., recency_half_life=120.,
                          burden_window=240., burden_half_life=120., cooldown=30., cap=4, kappa=.02, tau=.05)
FIXTURE_PRIORS = {f"C{j}": [1., 1.] for j in range(4)}
FIXTURE_PRIORS.update({f"D{j}": [1., 18., 1.] if j == 2 else [9., 1., 10.] for j in range(4)})
PERMUTATION_SEED = 314159
BOOTSTRAP_SEED = 271828
