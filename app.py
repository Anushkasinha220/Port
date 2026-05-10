from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sim.port_env import PortEnvironment, fixed_baseline_policy, run_episode
from src.engine import AgentConfig, GranularQLearningAgent
from src.mlops import log_experiment


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "port_optimus_q_agent.pkl"


st.set_page_config(
    page_title="Port-Optimus",
    page_icon="P",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    :root {
        --bg: #eef2f6;
        --panel: #ffffff;
        --panel-2: #f8fafc;
        --ink: #0f172a;
        --muted: #64748b;
        --border: #d8dee8;
        --teal: #0f766e;
        --teal-soft: #ccfbf1;
        --amber: #b45309;
        --amber-soft: #fef3c7;
        --red: #b91c1c;
        --red-soft: #fee2e2;
        --slate: #111827;
    }

    .stApp {
        background: var(--bg);
        color: var(--ink);
    }

    .block-container {
        padding-top: 1.35rem;
        padding-bottom: 2rem;
        max-width: 1540px;
    }

    [data-testid="stSidebar"] {
        background: #0b1220;
        border-right: 1px solid #1f2937;
    }

    [data-testid="stSidebar"] * {
        color: #f8fafc;
    }

    [data-testid="stSidebar"] .stButton > button {
        background: #111827;
        border: 1px solid #334155;
        border-radius: 8px;
        color: #f8fafc;
        font-weight: 700;
        min-height: 42px;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        background: #1e293b;
        border-color: #2dd4bf;
        color: #ffffff;
    }

    [data-testid="stSidebar"] .stSlider,
    [data-testid="stSidebar"] .stCheckbox {
        background: transparent;
    }

    .brand-block {
        padding: 4px 0 18px;
        border-bottom: 1px solid #273449;
        margin-bottom: 14px;
    }

    .brand-mark {
        width: 36px;
        height: 36px;
        border-radius: 8px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin-right: 9px;
        background: #14b8a6;
        color: #042f2e;
        font-weight: 900;
    }

    .brand-title {
        display: inline-block;
        vertical-align: top;
        color: #f8fafc;
        font-size: 20px;
        font-weight: 850;
        line-height: 1.05;
    }

    .brand-subtitle {
        display: block;
        color: #94a3b8;
        font-size: 12px;
        font-weight: 650;
        margin-top: 3px;
    }

    .command-header {
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto;
        gap: 18px;
        align-items: end;
        margin-bottom: 16px;
    }

    .eyeline {
        color: var(--teal);
        font-size: 12px;
        font-weight: 850;
        letter-spacing: .08em;
        text-transform: uppercase;
        margin-bottom: 5px;
    }

    .command-title {
        color: var(--ink);
        font-size: 34px;
        font-weight: 850;
        line-height: 1.08;
        margin: 0;
    }

    .command-copy {
        color: var(--muted);
        font-size: 14px;
        line-height: 1.45;
        max-width: 760px;
        margin-top: 7px;
    }

    .run-badge {
        background: #0f172a;
        border: 1px solid #263244;
        border-radius: 8px;
        color: #e2e8f0;
        min-width: 255px;
        padding: 13px 14px;
        text-align: left;
    }

    .run-badge span {
        color: #94a3b8;
        display: block;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: .07em;
        text-transform: uppercase;
    }

    .run-badge strong {
        color: #ffffff;
        display: block;
        font-size: 16px;
        margin-top: 4px;
    }

    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 12px;
        margin: 8px 0 14px;
    }

    .kpi {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 15px 16px 14px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, .045);
        min-height: 104px;
    }

    .kpi-label {
        color: var(--muted);
        display: flex;
        justify-content: space-between;
        gap: 8px;
        font-size: 11px;
        font-weight: 850;
        letter-spacing: .06em;
        text-transform: uppercase;
    }

    .kpi-value {
        color: var(--ink);
        font-size: 29px;
        font-weight: 880;
        line-height: 1.1;
        margin-top: 10px;
    }

    .kpi-note {
        color: var(--muted);
        font-size: 12px;
        font-weight: 650;
        margin-top: 8px;
    }

    .panel {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 8px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, .045);
        margin-bottom: 14px;
        padding: 17px;
    }

    .panel-title {
        color: var(--ink);
        font-size: 16px;
        font-weight: 850;
        margin-bottom: 4px;
    }

    .panel-subtitle {
        color: var(--muted);
        font-size: 12px;
        font-weight: 650;
        margin-bottom: 14px;
    }

    .port-map {
        background:
            linear-gradient(90deg, rgba(15, 118, 110, .08) 1px, transparent 1px),
            linear-gradient(0deg, rgba(15, 118, 110, .08) 1px, transparent 1px),
            #f8fafc;
        background-size: 36px 36px;
        border: 1px solid #dbe3ee;
        border-radius: 8px;
        padding: 14px 16px;
    }

    .lane {
        display: grid;
        grid-template-columns: 142px minmax(0, 1fr) 84px;
        gap: 14px;
        align-items: center;
        border-bottom: 1px solid #e2e8f0;
        min-height: 74px;
        padding: 12px 0;
    }

    .lane:last-child {
        border-bottom: 0;
    }

    .lane-name {
        color: #334155;
        font-size: 12px;
        font-weight: 850;
        letter-spacing: .07em;
        text-transform: uppercase;
    }

    .lane-asset {
        color: #0f172a;
        font-size: 25px;
        min-height: 34px;
        overflow: hidden;
        white-space: nowrap;
    }

    .lane-score {
        color: var(--muted);
        font-size: 12px;
        font-weight: 800;
        text-align: right;
    }

    .battery-track {
        background: #e2e8f0;
        border-radius: 999px;
        height: 11px;
        margin-top: 10px;
        overflow: hidden;
    }

    .battery-fill {
        background: linear-gradient(90deg, #0f766e, #14b8a6);
        height: 100%;
    }

    .frame-footer {
        align-items: center;
        display: flex;
        flex-wrap: wrap;
        gap: 9px;
        margin-top: 13px;
    }

    .chip {
        border-radius: 999px;
        display: inline-flex;
        align-items: center;
        min-height: 28px;
        padding: 5px 10px;
        font-size: 12px;
        font-weight: 800;
        border: 1px solid #cbd5e1;
        background: #ffffff;
        color: #334155;
    }

    .chip.teal {
        background: var(--teal-soft);
        border-color: #5eead4;
        color: #115e59;
    }

    .chip.amber {
        background: var(--amber-soft);
        border-color: #f59e0b;
        color: #92400e;
    }

    .chip.red {
        background: var(--red-soft);
        border-color: #ef4444;
        color: #991b1b;
    }

    .decision {
        background: #0f172a;
        border-radius: 8px;
        color: #e2e8f0;
        padding: 15px;
    }

    .decision-label {
        color: #94a3b8;
        font-size: 11px;
        font-weight: 850;
        letter-spacing: .07em;
        text-transform: uppercase;
    }

    .decision-action {
        color: #ffffff;
        font-size: 25px;
        font-weight: 880;
        line-height: 1.12;
        margin: 8px 0 10px;
    }

    .decision-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
        margin-top: 12px;
    }

    .decision-stat {
        background: #182235;
        border: 1px solid #263244;
        border-radius: 8px;
        padding: 10px;
    }

    .decision-stat span {
        color: #94a3b8;
        display: block;
        font-size: 11px;
        font-weight: 800;
    }

    .decision-stat strong {
        color: #f8fafc;
        display: block;
        font-size: 18px;
        margin-top: 3px;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 8px;
        overflow: hidden;
    }

    .stButton > button {
        border-radius: 8px;
        font-weight: 800;
        min-height: 42px;
    }

    @media (max-width: 1100px) {
        .command-header { grid-template-columns: 1fr; }
        .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .lane { grid-template-columns: 1fr; gap: 5px; }
        .lane-score { text-align: left; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_or_train_agent(storm: bool = False, peak_season: bool = False) -> GranularQLearningAgent:
    if MODEL_PATH.exists() and not storm and not peak_season:
        return GranularQLearningAgent.load_model(MODEL_PATH)

    config = AgentConfig(episodes=220 if not (storm or peak_season) else 120)
    agent = GranularQLearningAgent(config=config, seed=2026)
    agent.train(
        lambda: PortEnvironment(storm=storm, peak_season=peak_season),
        episodes=config.episodes,
    )
    if not storm and not peak_season:
        agent.save_model(MODEL_PATH)
    return agent


def simulate_agent(agent: GranularQLearningAgent, storm: bool, peak_season: bool) -> Tuple[List[Dict], Dict[str, float]]:
    env = PortEnvironment(seed=7, storm=storm, peak_season=peak_season)
    state = env.reset()
    done = False
    while not done:
        action = agent.predict(state)
        state, _, done, _ = env.step(action)
    return env.history, env.metrics()


def simulate_baseline(storm: bool, peak_season: bool) -> Tuple[List[Dict], Dict[str, float]]:
    env = PortEnvironment(seed=7, storm=storm, peak_season=peak_season)
    result = run_episode(env, fixed_baseline_policy)
    return result["history"], result["metrics"]


def normalize_scores(agent_metrics: Dict[str, float], baseline_metrics: Dict[str, float]) -> Tuple[List[float], List[float]]:
    def score(row: Dict[str, float]) -> List[float]:
        speed = min(100, row["Throughput_Rate"] / 4.8 * 100)
        cost = max(0, 100 - row["Total_Cost"] / 5.2)
        energy = min(100, row["Energy_Efficiency_Ratio"] / 0.36 * 100)
        safety = min(100, row["Average_Safety"])
        return [speed, cost, energy, safety]

    return score(agent_metrics), score(baseline_metrics)


def chart_template(fig: go.Figure, height: int, title: str) -> go.Figure:
    fig.update_layout(
        height=height,
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=16, color="#0f172a")),
        margin=dict(l=28, r=26, t=50, b=28),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#334155", family="Arial, sans-serif", size=12),
        legend=dict(orientation="h", y=-0.12, x=0),
    )
    return fig


def radar_chart(agent_metrics: Dict[str, float], baseline_metrics: Dict[str, float]) -> go.Figure:
    categories = ["Speed", "Cost", "Energy", "Safety"]
    agent_scores, baseline_scores = normalize_scores(agent_metrics, baseline_metrics)
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=agent_scores + [agent_scores[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name="RL Agent",
            line=dict(color="#0f766e", width=3),
            fillcolor="rgba(15, 118, 110, .18)",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=baseline_scores + [baseline_scores[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name="Human-Fixed Baseline",
            line=dict(color="#f59e0b", width=3),
            fillcolor="rgba(245, 158, 11, .14)",
        )
    )
    fig.update_layout(
        polar=dict(
            bgcolor="#ffffff",
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#e2e8f0"),
            angularaxis=dict(gridcolor="#e2e8f0"),
        )
    )
    return chart_template(fig, 356, "RL Agent vs Human Baseline")


def energy_heatmap(history: List[Dict]) -> go.Figure:
    df = pd.DataFrame(history)
    hours = list(range(24))
    energy = [0.0] * 24
    for _, row in df.iterrows():
        energy[int(row["hour"])] = float(row["energy_used"])
    fig = go.Figure(
        data=go.Heatmap(
            z=[energy],
            x=[f"{hour:02d}" for hour in hours],
            y=["Energy"],
            colorscale=[
                [0, "#f8fafc"],
                [0.42, "#5eead4"],
                [0.75, "#f59e0b"],
                [1, "#b91c1c"],
            ],
            colorbar=dict(title="kWh"),
            hovertemplate="Hour %{x}:00<br>Energy %{z:.2f} kWh<extra></extra>",
        )
    )
    return chart_template(fig, 260, "24-Hour Energy Usage")


def price_curve(history: List[Dict]) -> go.Figure:
    df = pd.DataFrame(history)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["hour"],
            y=df["electricity_price"],
            name="Price",
            mode="lines+markers",
            line=dict(color="#0f766e", width=3),
            marker=dict(size=6),
            fill="tozeroy",
            fillcolor="rgba(15, 118, 110, .10)",
            hovertemplate="Hour %{x}:00<br>$%{y:.2f}/kWh<extra></extra>",
        )
    )
    fig.update_xaxes(title="", tickmode="linear", dtick=3, gridcolor="#e2e8f0")
    fig.update_yaxes(title="", gridcolor="#e2e8f0")
    return chart_template(fig, 226, "Electricity Price Curve")


def kpi_strip(history: List[Dict], metrics: Dict[str, float], stress_label: str) -> None:
    latest = history[-1]
    queue = int(latest["queue_length"])
    battery = int(latest["battery_level"])
    congestion = float(latest["congestion"])
    risk_text = "High" if congestion >= 0.65 else ("Elevated" if congestion >= 0.38 else "Stable")
    risk_class = "red" if congestion >= 0.65 else ("amber" if congestion >= 0.38 else "teal")
    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="kpi">
                <div class="kpi-label"><span>Scenario</span><span>{stress_label}</span></div>
                <div class="kpi-value">{queue}</div>
                <div class="kpi-note">containers in active queue</div>
            </div>
            <div class="kpi">
                <div class="kpi-label"><span>Port Battery</span><span>{battery}%</span></div>
                <div class="kpi-value">{battery}%</div>
                <div class="battery-track"><div class="battery-fill" style="width:{battery}%"></div></div>
            </div>
            <div class="kpi">
                <div class="kpi-label"><span>Throughput</span><span>per hour</span></div>
                <div class="kpi-value">{metrics['Throughput_Rate']:.2f}</div>
                <div class="kpi-note">{metrics['Total_Processed']:.0f} units processed</div>
            </div>
            <div class="kpi">
                <div class="kpi-label"><span>Energy Ratio</span><span>units/kWh</span></div>
                <div class="kpi-value">{metrics['Energy_Efficiency_Ratio']:.2f}</div>
                <div class="kpi-note">${metrics['Total_Cost']:.1f} grid cost</div>
            </div>
            <div class="kpi">
                <div class="kpi-label"><span>Risk Posture</span><span class="chip {risk_class}">{risk_text}</span></div>
                <div class="kpi-value">{metrics['Average_Safety']:.0f}</div>
                <div class="kpi-note">average safety score</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def live_frame(row: Dict, lanes: Dict[str, str]) -> None:
    battery = max(0, min(100, int(row["battery_level"])))
    congestion = float(row["congestion"])
    risk_class = "red" if congestion >= 0.65 else ("amber" if congestion >= 0.38 else "teal")
    st.markdown(
        f"""
        <div class="panel">
            <div class="panel-title">Live Port Digital Twin</div>
            <div class="panel-subtitle">Berth queue, container buffer, truck dispatch lane, and battery rail.</div>
            <div class="port-map">
                <div class="lane">
                    <div class="lane-name">Berth Queue</div>
                    <div class="lane-asset">{lanes['ships']}</div>
                    <div class="lane-score">{int(row['queue_length'])} waiting</div>
                </div>
                <div class="lane">
                    <div class="lane-name">Buffer Grid</div>
                    <div class="lane-asset">{lanes['containers']}</div>
                    <div class="lane-score">{int(row['units_processed'])} moved</div>
                </div>
                <div class="lane">
                    <div class="lane-name">E-Truck Lane</div>
                    <div class="lane-asset">{lanes['trucks']}</div>
                    <div class="lane-score">{int(row['truck_availability'])} ready</div>
                </div>
                <div class="battery-track"><div class="battery-fill" style="width:{battery}%"></div></div>
                <div class="frame-footer">
                    <span class="chip teal">{row['action_name']}</span>
                    <span class="chip">Hour {int(row['hour']):02d}:00</span>
                    <span class="chip">Battery {battery}%</span>
                    <span class="chip">Price ${row['electricity_price']:.2f}/kWh</span>
                    <span class="chip {risk_class}">Congestion {row['congestion']:.2f}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def decision_panel(row: Dict, metrics: Dict[str, float]) -> None:
    congestion = float(row["congestion"])
    risk_label = "Critical" if congestion >= 0.65 else ("Watch" if congestion >= 0.38 else "Nominal")
    risk_class = "red" if congestion >= 0.65 else ("amber" if congestion >= 0.38 else "teal")
    st.markdown(
        f"""
        <div class="panel">
            <div class="panel-title">Policy Decision Engine</div>
            <div class="panel-subtitle">Current action selected by the learned RL control policy.</div>
            <div class="decision">
                <div class="decision-label">Recommended Action</div>
                <div class="decision-action">{row['action_name']}</div>
                <span class="chip {risk_class}">{risk_label} congestion risk</span>
                <div class="decision-grid">
                    <div class="decision-stat"><span>Energy Used</span><strong>{row['energy_used']:.1f}</strong></div>
                    <div class="decision-stat"><span>Grid Cost</span><strong>${row['energy_cost']:.1f}</strong></div>
                    <div class="decision-stat"><span>Wait Penalty</span><strong>{row['wait_penalty']:.1f}</strong></div>
                    <div class="decision-stat"><span>Run Safety</span><strong>{metrics['Average_Safety']:.0f}</strong></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def run_live_animation(
    agent: GranularQLearningAgent,
    storm: bool,
    peak_season: bool,
    frame_delay: float,
) -> Tuple[List[Dict], Dict[str, float]]:
    env = PortEnvironment(seed=12, storm=storm, peak_season=peak_season)
    state = env.reset()
    frame = st.empty()
    done = False
    while not done:
        action = agent.predict(state)
        state, _, done, row = env.step(action)
        with frame.container():
            live_frame(row, env.render_lanes())
        time.sleep(frame_delay)
    return env.history, env.metrics()


if "storm" not in st.session_state:
    st.session_state.storm = False
if "peak_season" not in st.session_state:
    st.session_state.peak_season = False

st.sidebar.markdown(
    """
    <div class="brand-block">
        <span class="brand-mark">P</span>
        <span class="brand-title">Port-Optimus<span class="brand-subtitle">Autonomous logistics grid</span></span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.caption("Stress testing")

if st.sidebar.button("Trigger Storm", use_container_width=True):
    st.session_state.storm = True
if st.sidebar.button("Peak Season", use_container_width=True):
    st.session_state.peak_season = True
if st.sidebar.button("Reset Stress Tests", use_container_width=True):
    st.session_state.storm = False
    st.session_state.peak_season = False

animation_speed = st.sidebar.slider("Frame delay", 0.05, 0.5, 0.16, 0.01)
log_runs = st.sidebar.toggle("Log simulation runs", value=True)

stress_label = "Storm + Peak" if st.session_state.storm and st.session_state.peak_season else (
    "Storm" if st.session_state.storm else ("Peak Season" if st.session_state.peak_season else "Normal")
)

with st.spinner("Loading RL control policy..."):
    agent = load_or_train_agent(st.session_state.storm, st.session_state.peak_season)
agent_history, agent_metrics = simulate_agent(agent, st.session_state.storm, st.session_state.peak_season)
baseline_history, baseline_metrics = simulate_baseline(st.session_state.storm, st.session_state.peak_season)
latest_row = agent_history[min(5, len(agent_history) - 1)]

st.markdown(
    f"""
    <div class="command-header">
        <div>
            <div class="eyeline">Autonomous port command center</div>
            <h1 class="command-title">Port-Optimus Logistics & Energy Grid</h1>
            <div class="command-copy">
                RL-driven container routing under congestion, limited battery capacity, truck availability,
                and volatile electricity prices.
            </div>
        </div>
        <div class="run-badge">
            <span>Control Session</span>
            <strong>{datetime.now().strftime('%d %b %Y, %H:%M')} | {stress_label}</strong>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

kpi_strip(agent_history, agent_metrics, stress_label)

left, right = st.columns([1.55, 1.0], gap="large")
with left:
    if st.button("Run Live Simulation", type="primary", use_container_width=True):
        agent_history, agent_metrics = run_live_animation(
            agent,
            st.session_state.storm,
            st.session_state.peak_season,
            animation_speed,
        )
        latest_row = agent_history[-1]
        if log_runs:
            run = log_experiment(
                params={
                    "agent": "GranularQLearningAgent",
                    "stress_test": stress_label,
                    "storm": st.session_state.storm,
                    "peak_season": st.session_state.peak_season,
                    "actions": "0 Rapid Discharge, 1 Buffered Move, 2 Idle/Charge",
                },
                metrics=agent_metrics,
                model_path=str(MODEL_PATH if MODEL_PATH.exists() else "session-trained"),
                stress_test=stress_label,
            )
            st.success(f"Logged run {run['run_id']} to experiments/metadata.json and results.csv")
    else:
        preview_env = PortEnvironment(seed=12, storm=st.session_state.storm, peak_season=st.session_state.peak_season)
        preview_env.queue_length = int(latest_row["queue_length"])
        preview_env.truck_availability = int(latest_row["truck_availability"])
        live_frame(latest_row, preview_env.render_lanes())

with right:
    decision_panel(latest_row, agent_metrics)
    st.plotly_chart(price_curve(agent_history), use_container_width=True)

analytics_left, analytics_right = st.columns([1.0, 1.0], gap="large")
with analytics_left:
    st.plotly_chart(radar_chart(agent_metrics, baseline_metrics), use_container_width=True)
with analytics_right:
    st.plotly_chart(energy_heatmap(agent_history), use_container_width=True)

table_left, table_right = st.columns([1.0, 1.25], gap="large")
with table_left:
    st.markdown('<div class="panel-title">RL vs Human-Fixed Baseline</div>', unsafe_allow_html=True)
    comparison = pd.DataFrame(
        [
            {"Controller": "RL Agent", **agent_metrics},
            {"Controller": "Human-Fixed Baseline", **baseline_metrics},
        ]
    )
    st.dataframe(
        comparison[
            [
                "Controller",
                "Energy_Efficiency_Ratio",
                "Throughput_Rate",
                "Total_Cost",
                "Average_Safety",
            ]
        ],
        hide_index=True,
        use_container_width=True,
        height=112,
    )

with table_right:
    st.markdown('<div class="panel-title">Control Policy Trace</div>', unsafe_allow_html=True)
    trace = pd.DataFrame(agent_history)[
        [
            "hour",
            "action_name",
            "queue_length",
            "battery_level",
            "electricity_price",
            "energy_used",
            "units_processed",
            "congestion",
        ]
    ]
    st.dataframe(trace, hide_index=True, use_container_width=True, height=260)
