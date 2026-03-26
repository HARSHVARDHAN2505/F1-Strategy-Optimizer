"""
F1 Strategy Engine — Interactive Streamlit Dashboard
=====================================================
Run with:
    streamlit run frontend/streamlit_app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

from backend.simulator import (
    load_model, simulate_race, find_best_one_stop,
    find_best_two_stop, monte_carlo_simulation,
    driver_comparison, RACES_2026, DRIVERS_2026,
)

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="F1 Strategy Engine",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark theme CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background-color: #0e1117; }
.metric-card {
    background: #1c1f26; border: 1px solid #2d3139;
    border-radius: 12px; padding: 1rem 1.25rem; text-align: center;
}
.metric-label { font-size: 11px; color: #6b7280; text-transform: uppercase; letter-spacing: .08em; }
.metric-value { font-size: 26px; font-weight: 700; margin-top: 4px; }
.metric-sub   { font-size: 11px; color: #6b7280; margin-top: 2px; }
.rec-box {
    background: #1c1f26; border-left: 4px solid #e10600;
    border-radius: 0; padding: 1rem 1.25rem; margin-bottom: 1rem;
}
h1, h2, h3 { color: #f9fafb !important; }
</style>
""", unsafe_allow_html=True)


# ── Load model (runs once, cached forever) ────────────────────────────────
@st.cache_resource
def get_model():
    try:
        return load_model()
    except FileNotFoundError:
        st.error("Model not found. Run: python train_model.py")
        st.stop()

model, encoder = get_model()


# ── ALL cached computation functions ─────────────────────────────────────
# These only rerun when their input arguments change.
# Switching tabs, scrolling, or any UI interaction won't trigger them.

@st.cache_data
def run_optimization(start_c, pit_loss, total_laps, base_off, max_stops):
    r1 = find_best_one_stop(
        model, encoder, start_c, pit_loss,
        total_laps, race_base_offset=base_off
    )
    r2 = (
        find_best_two_stop(
            model, encoder, start_c, pit_loss,
            total_laps, step=4, race_base_offset=base_off
        )
        if max_stops >= 2
        else {"best_time": float("inf"), "best_laps": [], "all_results": {}}
    )
    return r1, r2


@st.cache_data
def run_driver_comparison(race_name, start_c, pit_loss):
    return driver_comparison(
        model, encoder,
        race_name=race_name,
        start_compound=start_c,
        pit_stop_loss=pit_loss,
    )


@st.cache_data
def run_monte_carlo(start_c, rec_pits_tuple, pit_loss, total_laps, base_off, n_sims, noise):
    # rec_pits passed as tuple because lists aren't hashable for st.cache_data
    return monte_carlo_simulation(
        model, encoder, start_c, list(rec_pits_tuple),
        pit_loss, total_laps, n_sims, noise,
        race_base_offset=base_off,
    )


@st.cache_data
def run_heatmap(start_c, pit_loss, total_laps, base_off):
    compounds  = ["SOFT", "MEDIUM", "HARD"]
    pit_laps_h = list(range(10, total_laps - 5, 3))
    z = []
    for comp in compounds:
        row = []
        for pl in pit_laps_h:
            r = simulate_race(
                model, encoder, comp, [pl], pit_loss,
                total_laps, race_base_offset=base_off
            )
            row.append(round(r["total_time"], 1))
        z.append(row)
    return z, pit_laps_h


@st.cache_data
def run_degradation(start_c, stint, base_off):
    tyre_lives = list(range(1, stint + 1))
    result = {}
    for comp in ["SOFT", "MEDIUM", "HARD"]:
        times = []
        for tl in tyre_lives:
            r = simulate_race(
                model, encoder, comp, [], 0, tl,
                race_base_offset=base_off
            )
            times.append(r["lap_times"][-1])
        result[comp] = times
    return tyre_lives, result


@st.cache_data
def run_lap_breakdown(start_c, rec_pits_tuple, pit_loss, total_laps, base_off):
    return simulate_race(
        model, encoder, start_c, list(rec_pits_tuple),
        pit_loss, total_laps, race_base_offset=base_off
    )


# ── Helpers ───────────────────────────────────────────────────────────────
def fmt(s):
    m = int(s // 60)
    return f"{m}:{s % 60:05.2f}"

def metric_card(label, value, sub="", color="#f9fafb"):
    return f"""<div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color:{color}">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>"""


# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏎️ F1 Strategy Engine")
    st.markdown("---")

    st.subheader("2026 Season")
    race_names  = [f"R{r['round']} — {r['name']}" for r in RACES_2026]
    race_choice = st.selectbox("Select race", race_names, index=2)
    race_idx    = race_names.index(race_choice)
    selected_race = RACES_2026[race_idx]

    if selected_race.get("done"):
        st.success(f"Completed — {selected_race['winner']} won")

    st.markdown("---")
    st.subheader("Parameters")

    start_compound = st.selectbox("Starting compound", ["MEDIUM", "SOFT", "HARD"])
    total_laps     = selected_race["laps"]
    st.caption(f"Race laps: {total_laps}")

    pit_stop_loss = st.slider("Pit stop time loss (s)", 18.0, 32.0, 22.0, 0.5)
    max_stops     = st.radio("Max stops", [1, 2], index=1, horizontal=True)

    st.markdown("---")
    st.subheader("Monte Carlo")
    n_sims    = st.slider("Simulations", 100, 1000, 400, 100)
    noise_std = st.slider("Lap noise ±(s)", 0.1, 2.0, 0.4, 0.1)

    st.button("Run analysis", type="primary", use_container_width=True)


# ── Compute — only reruns when sidebar values actually change ─────────────
race_base_offset = selected_race["base"] - 72.5

r1, r2 = run_optimization(
    start_compound, pit_stop_loss, total_laps, race_base_offset, max_stops
)

times    = [r1["best_time"]]
if max_stops >= 2 and r2["best_time"] < float("inf"):
    times.append(r2["best_time"])
best_time = min(times)
rec       = "2-stop" if (max_stops >= 2 and r2["best_time"] < r1["best_time"]) else "1-stop"
gain      = max(times) - min(times)
rec_pits  = r2["best_laps"] if rec == "2-stop" else [r1["best_lap"]]

driver_results = run_driver_comparison(
    selected_race["name"], start_compound, pit_stop_loss
)


# ── Header ────────────────────────────────────────────────────────────────
st.markdown(f"## 🏎️ {selected_race['name']} GP — Strategy Analysis")
st.caption(
    f"Round {selected_race['round']} · {total_laps} laps · "
    f"{start_compound} start · {pit_stop_loss}s pit loss"
)
st.markdown("---")


# ── Tabs ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Strategy optimizer",
    "👥 Driver comparison",
    "🎲 Monte Carlo",
    "🔥 Heatmap",
    "🔴 Tyre degradation",
    "📄 Race report",
])


# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — STRATEGY OPTIMIZER
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    cols = st.columns(4)
    with cols[0]:
        st.markdown(metric_card(
            "Best 1-stop pit", f"Lap {r1['best_lap']}",
            fmt(r1["best_time"]), "#60a5fa"
        ), unsafe_allow_html=True)
    with cols[1]:
        laps_str = " & ".join(map(str, r2.get("best_laps", []))) if max_stops >= 2 else "—"
        st.markdown(metric_card(
            "Best 2-stop pits", laps_str,
            fmt(r2["best_time"]) if r2["best_time"] < float("inf") else "disabled",
            "#34d399"
        ), unsafe_allow_html=True)
    with cols[2]:
        st.markdown(metric_card(
            "Recommended", rec.upper(),
            f"Saves {gain:.1f}s", "#e10600"
        ), unsafe_allow_html=True)
    with cols[3]:
        st.markdown(metric_card(
            "Best race time", fmt(best_time),
            "optimal strategy", "#f9fafb"
        ), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if rec == "2-stop":
        rec_detail = (f"Pit on laps <b>{r2['best_laps'][0]}</b> and "
                      f"<b>{r2['best_laps'][1]}</b>. 2-stop benefits from "
                      f"fresher tyres in the final stint.")
    else:
        rec_detail = (f"Pit on lap <b>{r1['best_lap']}</b>. "
                      f"1-stop avoids extra pit lane time loss.")

    st.markdown(f"""<div class="rec-box">
        <b>Recommendation: {rec}</b><br>
        {rec_detail} Saves <b>{gain:.1f}s</b> vs the alternative.
    </div>""", unsafe_allow_html=True)

    # Strategy curve
    pit_laps_range = list(range(10, total_laps - 4))
    strat_times    = [r1["all_results"].get(l, r1["best_time"]) for l in pit_laps_range]
    mn, mx = min(strat_times), max(strat_times)
    pad    = (mx - mn) * 0.15 or 1

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pit_laps_range, y=strat_times, mode="lines",
        line=dict(color="#3b82f6", width=2),
        fill="tozeroy", fillcolor="rgba(59,130,246,0.07)"
    ))
    fig.add_vline(
        x=r1["best_lap"], line_dash="dash", line_color="#e10600",
        annotation_text=f"Optimal: Lap {r1['best_lap']}",
        annotation_position="top right"
    )
    fig.update_layout(
        template="plotly_dark", xaxis_title="Pit lap",
        yaxis_title="Total race time (s)", height=360,
        showlegend=False, yaxis_range=[mn - pad, mx + pad]
    )
    st.plotly_chart(fig, use_container_width=True)

    # Lap-by-lap breakdown (cached)
    lap_result = run_lap_breakdown(
        start_compound, tuple(rec_pits),
        pit_stop_loss, total_laps, race_base_offset
    )
    color_map  = {"SOFT": "#e10600", "MEDIUM": "#fbbf24", "HARD": "#9ca3af"}
    bar_colors = [color_map[c] for c in lap_result["compounds"]]
    fig2 = go.Figure(go.Bar(
        x=list(range(1, total_laps + 1)),
        y=lap_result["lap_times"],
        marker_color=bar_colors,
        hovertemplate="Lap %{x}: %{y:.2f}s"
    ))
    fig2.update_layout(
        template="plotly_dark", xaxis_title="Lap",
        yaxis_title="Lap time (s)", height=280,
        title="Lap-by-lap breakdown — optimal strategy"
    )
    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — DRIVER COMPARISON
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    # driver_results already computed above (cached)
    all_drivers      = [d["name"] for d in DRIVERS_2026]
    selected_drivers = st.multiselect(
        "Select drivers to compare",
        all_drivers,
        default=all_drivers[:10],
    )
    filtered = [r for r in driver_results if r["driver"] in selected_drivers]
    if not filtered:
        st.warning("Select at least one driver.")
        st.stop()

    leader_time = filtered[0]["race_time"]

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(metric_card(
            "Fastest", filtered[0]["driver"],
            fmt(leader_time), "#e10600"
        ), unsafe_allow_html=True)
    with m2:
        spread = filtered[-1]["race_time"] - leader_time
        st.markdown(metric_card(
            "Pace spread", f"{spread:.1f}s",
            "best to worst"
        ), unsafe_allow_html=True)
    with m3:
        n2 = sum(1 for r in filtered if r["strategy"] == "2-stop")
        st.markdown(metric_card(
            "On 2-stop", f"{n2} / {len(filtered)}",
            "drivers"
        ), unsafe_allow_html=True)
    with m4:
        avg_pit = round(np.mean([r["r1"]["best_lap"] for r in filtered]))
        st.markdown(metric_card(
            "Avg pit window", f"Lap {avg_pit}",
            "1-stop optimal"
        ), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    team_colors = {
        "McLaren": "#f97316", "Mercedes": "#22c55e", "Red Bull": "#3b82f6",
        "Ferrari": "#e10600", "Williams": "#8b5cf6", "Aston Martin": "#10b981",
        "Audi": "#6366f1", "Alpine": "#ec4899", "Racing Bulls": "#14b8a6",
        "Haas": "#f59e0b", "Cadillac": "#a3a3a3",
    }

    c1, c2 = st.columns(2)
    with c1:
        fig3 = go.Figure()
        for r in filtered:
            color = team_colors.get(r["team"], "#888")
            fig3.add_trace(go.Bar(
                name=r["driver"], x=[r["driver"]],
                y=[r["r1"]["best_time"]],
                marker_color=color + "66", showlegend=False
            ))
            fig3.add_trace(go.Bar(
                name=r["driver"], x=[r["driver"]],
                y=[r["r2"]["best_time"]],
                marker_color=color, showlegend=False
            ))
        fig3.update_layout(
            template="plotly_dark", barmode="group",
            yaxis_title="Race time (s)", height=320,
            title="1-stop (light) vs 2-stop (solid)"
        )
        st.plotly_chart(fig3, use_container_width=True)

    with c2:
        fig4 = go.Figure(go.Bar(
            x=[r["driver"] for r in filtered],
            y=[r["r1"]["best_lap"] for r in filtered],
            marker_color=[team_colors.get(r["team"], "#888") for r in filtered],
            text=[r["r1"]["best_lap"] for r in filtered],
            textposition="outside",
        ))
        fig4.update_layout(
            template="plotly_dark", yaxis_title="Optimal pit lap",
            height=320, title="Optimal 1-stop pit window by driver"
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("### Strategy ranking")
    df_rank = pd.DataFrame([{
        "Pos":        r["position"],
        "Driver":     r["driver"],
        "Team":       r["team"],
        "Strategy":   r["strategy"],
        "Pit laps":   " & ".join(map(str, r["pit_laps"])),
        "Race time":  fmt(r["race_time"]),
        "Gap":        f"+{r['gap']:.1f}s",
        "Pit window": f"L{r['r1']['best_lap']}–{min(r['r1']['best_lap']+4, total_laps-5)}",
    } for r in filtered])
    st.dataframe(df_rank, hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — MONTE CARLO
# ══════════════════════════════════════════════════════════════════════════
with tab3:
    mc = run_monte_carlo(
        start_compound, tuple(rec_pits),
        pit_stop_loss, total_laps, race_base_offset,
        n_sims, noise_std
    )

    mc_cols = st.columns(4)
    for col, label, val in zip(
        mc_cols,
        ["Mean", "Std dev", "Best case (P5)", "Worst case (P95)"],
        [mc["mean"], mc["std"], mc["p5"], mc["p95"]]
    ):
        with col:
            st.metric(label, fmt(val) if label != "Std dev" else f"{val:.2f}s")

    fig5 = px.histogram(
        mc["all_times"], nbins=40,
        template="plotly_dark",
        color_discrete_sequence=["#3b82f6"]
    )
    fig5.add_vline(
        x=mc["mean"], line_dash="dash", line_color="#e10600",
        annotation_text=f"Mean: {fmt(mc['mean'])}"
    )
    fig5.update_layout(
        xaxis_title="Total race time (s)", yaxis_title="Frequency",
        showlegend=False, height=360,
        title=f"Distribution of {n_sims} simulated race outcomes"
    )
    st.plotly_chart(fig5, use_container_width=True)

    st.info(
        f"**90% confidence interval:** {fmt(mc['p5'])} – {fmt(mc['p95'])} "
        f"| Range: {mc['p95'] - mc['p5']:.1f}s"
    )


# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — HEATMAP
# ══════════════════════════════════════════════════════════════════════════
with tab4:
    z, pit_laps_h = run_heatmap(
        start_compound, pit_stop_loss, total_laps, race_base_offset
    )
    compounds = ["SOFT", "MEDIUM", "HARD"]

    fig6 = go.Figure(go.Heatmap(
        z=z,
        x=[f"Lap {l}" for l in pit_laps_h],
        y=compounds,
        colorscale="RdYlGn_r",
        text=[[f"+{v - min([z[i][j] for i in range(3)]):.0f}s" for v in row] for row in z],
        texttemplate="%{text}",
    ))
    fig6.update_layout(
        template="plotly_dark", xaxis_title="Pit lap",
        yaxis_title="Starting compound", height=300,
        title="Race time delta vs fastest cell (green = faster)"
    )
    st.plotly_chart(fig6, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 5 — TYRE DEGRADATION
# ══════════════════════════════════════════════════════════════════════════
with tab5:
    stint = st.slider("Stint length", 10, 60, 40)

    tyre_lives, deg_data = run_degradation(start_compound, stint, race_base_offset)

    fig7 = go.Figure()
    deg_colors = {"SOFT": "#e10600", "MEDIUM": "#fbbf24", "HARD": "#9ca3af"}
    for comp, color in deg_colors.items():
        fig7.add_trace(go.Scatter(
            x=tyre_lives, y=deg_data[comp],
            mode="lines", name=comp,
            line=dict(color=color, width=2)
        ))
    fig7.update_layout(
        template="plotly_dark",
        xaxis_title="Tyre life (laps)",
        yaxis_title="Predicted lap time (s)",
        height=380,
        title="Tyre degradation — ML model prediction",
        legend=dict(orientation="h", y=1.05)
    )
    st.plotly_chart(fig7, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 6 — RACE REPORT
# ══════════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("Race strategy report")

    col_exp1, col_exp2, _ = st.columns([1, 1, 2])

    with col_exp1:
        rows = [["Pos", "Driver", "Team", "Strategy",
                 "Pit Laps", "Race Time (s)", "Race Time", "Gap (s)"]]
        for r in driver_results:
            rows.append([
                r["position"], r["driver"], r["team"], r["strategy"],
                " & ".join(map(str, r["pit_laps"])),
                f"{r['race_time']:.2f}", fmt(r["race_time"]),
                f"{r['gap']:.2f}"
            ])
        csv = "\n".join(",".join(map(str, row)) for row in rows)
        st.download_button(
            "⬇ Export CSV", csv,
            file_name=f"f1_2026_{selected_race['name'].lower()}_strategy.csv",
            mime="text/csv"
        )

    with col_exp2:
        report = {
            "race":          selected_race["name"] + " GP",
            "round":         selected_race["round"],
            "laps":          total_laps,
            "startCompound": start_compound,
            "pitLoss":       pit_stop_loss,
            "generatedAt":   datetime.now().isoformat(),
            "drivers": [{
                "position": r["position"],
                "name":     r["driver"],
                "team":     r["team"],
                "strategy": r["strategy"],
                "pitLaps":  r["pit_laps"],
                "raceTime": round(r["race_time"], 3),
                "gap":      r["gap"],
            } for r in driver_results],
            "monteCarlo": {
                "simulations": n_sims,
                "mean":        round(mc["mean"], 3),
                "std":         round(mc["std"], 3),
                "p5":          round(mc["p5"], 3),
                "p95":         round(mc["p95"], 3),
            },
        }
        st.download_button(
            "⬇ Export JSON", json.dumps(report, indent=2),
            file_name=f"f1_2026_{selected_race['name'].lower()}_strategy.json",
            mime="application/json"
        )

    st.markdown("---")
    rc1, rc2, rc3 = st.columns(3)

    with rc1:
        st.markdown("**Race info**")
        st.table(pd.DataFrame([
            ["Race",       selected_race["name"] + " GP"],
            ["Round",      selected_race["round"]],
            ["Laps",       total_laps],
            ["Start tyre", start_compound],
            ["Pit loss",   f"{pit_stop_loss}s"],
        ], columns=["", ""]))

    with rc2:
        st.markdown("**Top prediction**")
        top = driver_results[0]
        st.table(pd.DataFrame([
            ["P1",       top["driver"]],
            ["Team",     top["team"]],
            ["Strategy", top["strategy"]],
            ["Pit laps", " & ".join(map(str, top["pit_laps"]))],
            ["Time",     fmt(top["race_time"])],
        ], columns=["", ""]))

    with rc3:
        st.markdown("**Field summary**")
        avg = np.mean([r["race_time"] for r in driver_results])
        st.table(pd.DataFrame([
            ["Drivers",     len(driver_results)],
            ["Pace spread", f"{driver_results[-1]['race_time'] - driver_results[0]['race_time']:.1f}s"],
            ["Avg time",    fmt(avg)],
            ["On 1-stop",   sum(1 for r in driver_results if r["strategy"] == "1-stop")],
            ["On 2-stop",   sum(1 for r in driver_results if r["strategy"] == "2-stop")],
        ], columns=["", ""]))

    top = driver_results[0]
    sec = driver_results[1]
    st.info(
        f"**AI strategy call:** {top['driver']} ({top['team']}) leads at "
        f"{selected_race['name']} GP with a predicted time of {fmt(top['race_time'])} "
        f"on a **{top['strategy']}** strategy "
        f"(pit laps: {' & '.join(map(str, top['pit_laps']))}). "
        f"{sec['driver']} is the closest challenger at +{sec['gap']:.1f}s. "
        f"Monte Carlo shows a 90% CI of {mc['p95'] - mc['p5']:.1f}s — "
        f"{'tight, strategy is robust.' if mc['p95'] - mc['p5'] < 15 else 'wide, strategy sensitive to race chaos.'}"
    )