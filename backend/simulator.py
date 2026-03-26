"""
F1 Strategy Engine — Layer 3 & 4: Simulator + Optimizer
========================================================
HOW THE SIMULATION WORKS:
  simulate_race() is the core engine. It runs a for-loop over every
  lap of the race and calls the ML model to predict each lap time.

  The loop tracks:
    - tyre_life  : how many laps on this set (resets to 0 after pit)
    - compound   : current tyre type (changes at pit stop)
    - total_time : accumulates every lap time + pit stop costs

HOW THE OPTIMIZER WORKS:
  find_best_one_stop() and find_best_two_stop() are brute-force searches.
  There is no fancy algorithm — they just call simulate_race() for every
  possible pit lap combination and return the minimum.

  For 1-stop: ~50 simulations (one per possible pit lap 10→60)
  For 2-stop: ~1000 simulations (all (lap1, lap2) pairs)

  This is fast because each simulation is just arithmetic in a loop.
"""

import os
import pickle
import numpy as np
from itertools import product


# ── Compound rotation after a pit stop ────────────────────────────────────
# Simplified rule: SOFT→MEDIUM, MEDIUM→HARD, HARD→MEDIUM
# Real F1 has more complex tyre rules, but this captures the core trade-off
NEXT_COMPOUND = {
    "SOFT":   "MEDIUM",
    "MEDIUM": "HARD",
    "HARD":   "MEDIUM",
}

# ── 2026 F1 Calendar ──────────────────────────────────────────────────────
RACES_2026 = [
    {"round": 1,  "name": "Australia",   "laps": 58, "base": 83.5,  "done": True,  "winner": "Russell"},
    {"round": 2,  "name": "China",       "laps": 56, "base": 96.2,  "done": True,  "winner": "Antonelli"},
    {"round": 3,  "name": "Japan",       "laps": 53, "base": 92.0,  "done": False},
    {"round": 4,  "name": "Miami",       "laps": 57, "base": 90.3,  "done": False},
    {"round": 5,  "name": "Canada",      "laps": 70, "base": 75.1,  "done": False},
    {"round": 6,  "name": "Monaco",      "laps": 78, "base": 74.8,  "done": False},
    {"round": 7,  "name": "Barcelona",   "laps": 66, "base": 82.4,  "done": False},
    {"round": 8,  "name": "Austria",     "laps": 71, "base": 67.5,  "done": False},
    {"round": 9,  "name": "Britain",     "laps": 52, "base": 89.7,  "done": False},
    {"round": 10, "name": "Belgium",     "laps": 44, "base": 108.2, "done": False},
    {"round": 11, "name": "Hungary",     "laps": 70, "base": 79.8,  "done": False},
    {"round": 12, "name": "Netherlands", "laps": 72, "base": 74.6,  "done": False},
    {"round": 13, "name": "Italy",       "laps": 53, "base": 81.4,  "done": False},
    {"round": 14, "name": "Madrid",      "laps": 60, "base": 87.0,  "done": False},
    {"round": 15, "name": "Azerbaijan",  "laps": 51, "base": 105.6, "done": False},
    {"round": 16, "name": "Singapore",   "laps": 62, "base": 101.2, "done": False},
    {"round": 17, "name": "Austin",      "laps": 56, "base": 96.8,  "done": False},
    {"round": 18, "name": "Mexico",      "laps": 71, "base": 79.2,  "done": False},
    {"round": 19, "name": "Brazil",      "laps": 71, "base": 71.2,  "done": False},
    {"round": 20, "name": "Las Vegas",   "laps": 50, "base": 95.4,  "done": False},
    {"round": 21, "name": "Qatar",       "laps": 57, "base": 83.9,  "done": False},
    {"round": 22, "name": "Abu Dhabi",   "laps": 58, "base": 88.6,  "done": False},
]

# ── 2026 Driver Grid ───────────────────────────────────────────────────────
DRIVERS_2026 = [
    {"name": "Norris",     "team": "McLaren",       "pace": -0.42, "crew": 21.3},
    {"name": "Piastri",    "team": "McLaren",       "pace": -0.30, "crew": 21.4},
    {"name": "Russell",    "team": "Mercedes",      "pace": -0.28, "crew": 21.8},
    {"name": "Antonelli",  "team": "Mercedes",      "pace": -0.18, "crew": 21.9},
    {"name": "Verstappen", "team": "Red Bull",      "pace": -0.35, "crew": 21.5},
    {"name": "Hadjar",     "team": "Red Bull",      "pace":  0.08, "crew": 21.6},
    {"name": "Leclerc",    "team": "Ferrari",       "pace": -0.22, "crew": 21.7},
    {"name": "Hamilton",   "team": "Ferrari",       "pace": -0.14, "crew": 21.8},
    {"name": "Sainz",      "team": "Williams",      "pace": -0.05, "crew": 22.4},
    {"name": "Albon",      "team": "Williams",      "pace":  0.12, "crew": 22.5},
    {"name": "Alonso",     "team": "Aston Martin",  "pace":  0.06, "crew": 22.7},
    {"name": "Stroll",     "team": "Aston Martin",  "pace":  0.31, "crew": 22.8},
    {"name": "Hulkenberg", "team": "Audi",          "pace":  0.18, "crew": 23.1},
    {"name": "Bortoleto",  "team": "Audi",          "pace":  0.25, "crew": 23.2},
    {"name": "Gasly",      "team": "Alpine",        "pace":  0.20, "crew": 23.0},
    {"name": "Colapinto",  "team": "Alpine",        "pace":  0.35, "crew": 23.1},
    {"name": "Lawson",     "team": "Racing Bulls",  "pace":  0.22, "crew": 22.6},
    {"name": "Lindblad",   "team": "Racing Bulls",  "pace":  0.45, "crew": 22.7},
    {"name": "Ocon",       "team": "Haas",          "pace":  0.28, "crew": 23.3},
    {"name": "Bearman",    "team": "Haas",          "pace":  0.33, "crew": 23.4},
    {"name": "Perez",      "team": "Cadillac",      "pace":  0.15, "crew": 23.8},
    {"name": "Bottas",     "team": "Cadillac",      "pace":  0.22, "crew": 23.9},
]


def load_model(
    model_path="models/tire_model.pkl",
    encoder_path="models/compound_encoder.pkl",
):
    """Load the trained RandomForest and label encoder."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No model at {model_path}. Run: python train_model.py"
        )
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    encoder = None
    if os.path.exists(encoder_path):
        with open(encoder_path, "rb") as f:
            encoder = pickle.load(f)
    return model, encoder


def predict_lap_time(
    model, encoder, compound: str, tyre_life: int, lap_number: int,
    pace_modifier: float = 0.0, race_base_offset: float = 0.0,
) -> float:
    """
    Ask the ML model: given this tyre state, what is the lap time?

    pace_modifier    : driver-specific delta (e.g. -0.35 for Norris)
    race_base_offset : circuit-specific base time offset
    """
    compound_enc = (
        encoder.transform([compound])[0]
        if encoder is not None
        else ["HARD", "MEDIUM", "SOFT"].index(compound)
    )
    features = np.array([[lap_number, compound_enc, tyre_life]])
    pred = float(model.predict(features)[0])
    return pred + pace_modifier + race_base_offset


def simulate_race(
    model,
    encoder,
    start_compound: str,
    pit_laps: list,
    pit_stop_loss: float = 22.0,
    total_laps: int = 60,
    noise_std: float = 0.0,
    pace_modifier: float = 0.0,
    race_base_offset: float = 0.0,
) -> dict:
    """
    THE CORE LOOP — simulates an entire race lap by lap.

    Algorithm:
        total_time = 0
        tyre_life  = 0
        compound   = start_compound

        for lap in 1..total_laps:
            lap_time   = ML_model(compound, tyre_life, lap)
            lap_time  += random_noise  # only in Monte Carlo mode
            total_time += lap_time
            tyre_life  += 1

            if lap in pit_laps:
                total_time += pit_stop_loss   # time lost in pit lane
                tyre_life   = 0               # fresh tyres reset to 0
                compound    = next_compound   # e.g. MEDIUM → HARD

        return total_time

    Returns a dict with total_time, per-lap breakdown, and pit events.
    """
    total_time = 0.0
    tyre_life = 0
    compound = start_compound
    lap_times = []
    compounds = []
    tyre_lives = []
    events = []

    for lap in range(1, total_laps + 1):
        lap_time = predict_lap_time(
            model, encoder, compound, tyre_life, lap,
            pace_modifier, race_base_offset
        )

        # Monte Carlo mode: add random lap noise
        if noise_std > 0:
            lap_time += np.random.normal(0, noise_std)

        lap_time = max(lap_time, 55.0)   # physical minimum
        total_time += lap_time
        tyre_life += 1

        lap_times.append(round(lap_time, 3))
        compounds.append(compound)
        tyre_lives.append(tyre_life)

        if lap in pit_laps:
            old_compound = compound
            total_time += pit_stop_loss
            tyre_life = 0
            compound = NEXT_COMPOUND.get(compound, "MEDIUM")
            events.append({
                "lap": lap,
                "type": "pit",
                "from": old_compound,
                "to": compound,
                "loss": pit_stop_loss,
            })

    return {
        "total_time": round(total_time, 3),
        "lap_times":  lap_times,
        "compounds":  compounds,
        "tyre_lives": tyre_lives,
        "pit_laps":   pit_laps,
        "events":     events,
    }


def find_best_one_stop(
    model, encoder,
    start_compound: str = "MEDIUM",
    pit_stop_loss: float = 22.0,
    total_laps: int = 60,
    pace_modifier: float = 0.0,
    race_base_offset: float = 0.0,
) -> dict:
    """
    BRUTE-FORCE 1-STOP OPTIMIZER

    Tries every lap from 10 to (total_laps - 5) as the single pit stop.
    Runs simulate_race() for each. Returns the minimum.

    Why brute force?
      The search space is only ~50 options. At <1ms per simulation,
      the whole search completes in under 50ms. No need for anything
      fancier than a linear scan.
    """
    best_time = float("inf")
    best_lap = None
    all_results = {}

    for pit_lap in range(10, total_laps - 4):
        result = simulate_race(
            model, encoder, start_compound,
            pit_laps=[pit_lap],
            pit_stop_loss=pit_stop_loss,
            total_laps=total_laps,
            pace_modifier=pace_modifier,
            race_base_offset=race_base_offset,
        )
        all_results[pit_lap] = result["total_time"]
        if result["total_time"] < best_time:
            best_time = result["total_time"]
            best_lap = pit_lap

    return {"best_lap": best_lap, "best_time": best_time, "all_results": all_results}


def find_best_two_stop(
    model, encoder,
    start_compound: str = "MEDIUM",
    pit_stop_loss: float = 22.0,
    total_laps: int = 60,
    step: int = 4,
    pace_modifier: float = 0.0,
    race_base_offset: float = 0.0,
) -> dict:
    """
    BRUTE-FORCE 2-STOP OPTIMIZER

    Tries all (lap1, lap2) pairs where lap2 > lap1 + 10.
    The `step` parameter skips every other lap to speed up the search
    — step=2 halves the combinations while staying accurate within 2 laps.

    Complexity: O(n²) in number of laps, but n≈50 so ~625 simulations total.
    """
    best_time = float("inf")
    best_laps = None
    all_results = {}

    for lap1 in range(10, total_laps - 15, step):
        for lap2 in range(lap1 + 10, total_laps - 5, step):
            result = simulate_race(
                model, encoder, start_compound,
                pit_laps=[lap1, lap2],
                pit_stop_loss=pit_stop_loss,
                total_laps=total_laps,
                pace_modifier=pace_modifier,
                race_base_offset=race_base_offset,
            )
            all_results[(lap1, lap2)] = result["total_time"]
            if result["total_time"] < best_time:
                best_time = result["total_time"]
                best_laps = [lap1, lap2]

    return {"best_laps": best_laps, "best_time": best_time, "all_results": all_results}


def monte_carlo_simulation(
    model, encoder,
    start_compound: str,
    pit_laps: list,
    pit_stop_loss: float = 22.0,
    total_laps: int = 60,
    n_simulations: int = 500,
    noise_std: float = 0.4,
    pace_modifier: float = 0.0,
    race_base_offset: float = 0.0,
) -> dict:
    """
    MONTE CARLO UNCERTAINTY MODELING

    Runs the same race simulation n_simulations times, each time
    adding random noise to every lap time. This models:
      - Traffic and yellow flags
      - Driver errors
      - Weather variation
      - Mechanical degradation variance

    noise_std = 0.4s means each lap can vary ±0.4s randomly.
    Over 60 laps, this accumulates into a distribution of outcomes.

    The output histogram tells you:
      - Narrow peak → strategy is robust (works regardless of bad laps)
      - Wide spread → strategy is fragile (sensitive to race chaos)
    """
    all_times = []
    for _ in range(n_simulations):
        result = simulate_race(
            model, encoder, start_compound,
            pit_laps=pit_laps,
            pit_stop_loss=pit_stop_loss,
            total_laps=total_laps,
            noise_std=noise_std,
            pace_modifier=pace_modifier,
            race_base_offset=race_base_offset,
        )
        all_times.append(result["total_time"])

    all_times = np.array(all_times)
    return {
        "all_times": all_times.tolist(),
        "mean":      float(np.mean(all_times)),
        "std":       float(np.std(all_times)),
        "p5":        float(np.percentile(all_times, 5)),
        "p95":       float(np.percentile(all_times, 95)),
        "best":      float(np.min(all_times)),
        "worst":     float(np.max(all_times)),
    }


def compare_all_strategies(
    model, encoder,
    start_compound: str = "MEDIUM",
    pit_stop_loss: float = 22.0,
    total_laps: int = 60,
) -> dict:
    """
    Full strategy comparison: 1-stop vs 2-stop.
    Prints a summary and returns the recommendation.
    """
    print(f"\nComparing strategies ({start_compound} start, {pit_stop_loss}s pit loss, {total_laps} laps)...")
    one = find_best_one_stop(model, encoder, start_compound, pit_stop_loss, total_laps)
    two = find_best_two_stop(model, encoder, start_compound, pit_stop_loss, total_laps)

    recommended = "2-stop" if two["best_time"] < one["best_time"] else "1-stop"
    gain = abs(one["best_time"] - two["best_time"])

    print(f"\n  1-stop: lap {one['best_lap']}        → {one['best_time']:.1f}s")
    print(f"  2-stop: laps {two['best_laps']}  → {two['best_time']:.1f}s")
    print(f"  Recommended: {recommended} (saves {gain:.1f}s)")
    return {"one_stop": one, "two_stop": two, "recommended": recommended, "gain": round(gain, 2)}


def driver_comparison(
    model, encoder,
    race_name: str = "Monaco",
    start_compound: str = "MEDIUM",
    pit_stop_loss: float = 22.0,
) -> list:
    """
    Run strategy comparison for all 2026 drivers at a given race.
    Returns results sorted by predicted best race time.
    """
    race = next((r for r in RACES_2026 if r["name"] == race_name), RACES_2026[5])
    total_laps = race["laps"]
    base_offset = race["base"] - 72.5    # offset from default base

    results = []
    for driver in DRIVERS_2026:
        r1 = find_best_one_stop(
            model, encoder, start_compound, pit_stop_loss, total_laps,
            pace_modifier=driver["pace"], race_base_offset=base_offset,
        )
        r2 = find_best_two_stop(
            model, encoder, start_compound, pit_stop_loss, total_laps,
            pace_modifier=driver["pace"], race_base_offset=base_offset,
        )
        best = r1 if r1["best_time"] <= r2["best_time"] else r2
        results.append({
            "driver":    driver["name"],
            "team":      driver["team"],
            "strategy":  "1-stop" if r1["best_time"] <= r2["best_time"] else "2-stop",
            "pit_laps":  [r1["best_lap"]] if r1["best_time"] <= r2["best_time"] else r2["best_laps"],
            "race_time": best["best_time"] if "best_time" in best else best.get("best_time"),
            "r1":        r1,
            "r2":        r2,
        })

    results.sort(key=lambda x: x["race_time"])
    leader = results[0]["race_time"]
    for i, r in enumerate(results):
        r["position"] = i + 1
        r["gap"] = round(r["race_time"] - leader, 3)

    return results


if __name__ == "__main__":
    model, encoder = load_model()
    compare_all_strategies(model, encoder)