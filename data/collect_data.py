"""
F1 Strategy Engine — Data Collection
Pulls real lap telemetry from FastF1 API.
Run: python data/collect_data.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def collect(year=2023, gp="Monaco", session_type="R"):
    try:
        import fastf1
        fastf1.Cache.enable_cache("cache")
        print(f"Loading {year} {gp} GP ({session_type})...")
        session = fastf1.get_session(year, gp, session_type)
        session.load()

        laps = session.laps.pick_quicklaps()
        laps = laps[["LapNumber", "Compound", "TyreLife", "LapTime", "Driver"]].dropna()
        laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()
        laps = laps[(laps["LapTimeSeconds"] > 60) & (laps["LapTimeSeconds"] < 120)]
        laps = laps[["LapNumber", "Compound", "TyreLife", "LapTimeSeconds"]].reset_index(drop=True)

        os.makedirs("data", exist_ok=True)
        out = f"data/{year}_{gp.lower()}_laps.csv"
        laps.to_csv(out, index=False)
        print(f"Saved {len(laps)} laps → {out}")
        return laps

    except ImportError:
        print("FastF1 not installed. Run: pip install fastf1")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    collect()
