# app_preview_3tiers.py
# Preview dashboard (no backend): Synthetic traffic + Forecast baseline + 3-tier autoscaling + Events + Playback
# Run: streamlit run app_preview_3tiers.py

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# =========================
# Helpers
# =========================
def ceil_div(a: float, b: float) -> int:
    if b <= 0:
        return 0
    return int(math.ceil(a / b))


def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


# =========================
# Synthetic data
# =========================
def generate_synthetic_1m(
    minutes: int,
    seed: int,
    base_rps: float,
    trend_per_hour: float,
    season_amp: float,
    noise_sigma: float,
    spike_prob_per_hour: float,
    spike_duration_min_range: Tuple[int, int],
    spike_magnitude_range: Tuple[float, float],
    ddos_prob: float,
) -> pd.DataFrame:
    """
    Create 1-minute time series:
    - rps_true: demand
    - ddos-like features when spike occurs and ddos_prob triggers:
        top_country_share up, endpoint_entropy down, dir_nunique down, error_rate up
    """
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2025-01-01 00:00:00")
    ts = pd.date_range(t0, periods=minutes, freq="1min")

    # baseline = base + trend + daily-ish seasonality (sinus)
    # use 24h season if minutes cover multiple hours
    mins = np.arange(minutes)
    hours = mins / 60.0
    season = season_amp * (np.sin(2 * np.pi * hours / 24.0) + 0.2 * np.sin(2 * np.pi * hours / 12.0))
    trend = trend_per_hour * hours
    noise = rng.normal(0.0, noise_sigma, size=minutes)

    rps = np.maximum(0.1, base_rps + trend + season + noise)

    # inject spikes as bursts
    # spike_prob_per_hour: expected probability per hour; convert to per minute
    spike_prob_per_min = spike_prob_per_hour / 60.0

    spike_flag = np.zeros(minutes, dtype=bool)
    ddos_flag = np.zeros(minutes, dtype=bool)

    i = 0
    while i < minutes:
        if rng.random() < spike_prob_per_min:
            dur = int(rng.integers(spike_duration_min_range[0], spike_duration_min_range[1] + 1))
            mag = float(rng.uniform(spike_magnitude_range[0], spike_magnitude_range[1]))
            end = min(minutes, i + dur)
            # spike shape: ramp up then down (triangular-ish)
            span = end - i
            shape = 0.6 + 0.4 * np.sin(np.linspace(0, np.pi, span))  # length == span
            rps[i:end] = rps[i:end] * mag * shape
            spike_flag[i:end] = True

            # ddos-like subset of spikes
            if rng.random() < ddos_prob:
                ddos_flag[i:end] = True
            i = end
        else:
            i += 1

    # Build ddos-ish features (synthetic)
    # In normal times:
    top_country_share = rng.uniform(0.15, 0.35, size=minutes)
    endpoint_entropy = rng.uniform(2.5, 4.5, size=minutes)
    dir_nunique = rng.integers(50, 150, size=minutes)
    error_rate = rng.uniform(0.005, 0.03, size=minutes)

    # When ddos_flag: concentrated & more errors
    top_country_share[ddos_flag] = rng.uniform(0.65, 0.90, size=ddos_flag.sum())
    endpoint_entropy[ddos_flag] = rng.uniform(0.8, 1.8, size=ddos_flag.sum())
    dir_nunique[ddos_flag] = rng.integers(5, 25, size=ddos_flag.sum())
    error_rate[ddos_flag] = rng.uniform(0.05, 0.18, size=ddos_flag.sum())

    df = pd.DataFrame(
        {
            "timestamp": ts,
            "rps_true": rps,
            "spike_true": spike_flag,
            "ddos_true": ddos_flag,
            "top_country_share": top_country_share,
            "endpoint_entropy": endpoint_entropy,
            "dir_nunique": dir_nunique,
            "error_rate": error_rate,
        }
    )
    return df


# =========================
# Forecast baseline
# =========================
def add_baseline_forecasts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Demo forecast that "only catches trend":
    - pred_15m: longer rolling mean
    - pred_5m_mean: shorter rolling mean
    - pred_5m_q90: rolling quantile to mimic uncertainty bound
    - pred_1m (optional): rolling median (not used for spike catching)
    All are shifted by 1 to avoid leakage.
    """
    out = df.copy()
    s = out["rps_true"]

    out["pred_15m"] = s.rolling(120, min_periods=20).mean().shift(1).bfill()
    out["pred_5m_mean"] = s.rolling(30, min_periods=10).mean().shift(1).bfill()
    out["pred_5m_q90"] = s.rolling(30, min_periods=10).quantile(0.90).shift(1).bfill()
    out["pred_1m"] = s.rolling(10, min_periods=3).median().shift(1).bfill()

    return out


# =========================
# 3-tier policy + simulator
# =========================
@dataclass
class PolicyParams:
    cap_rps_per_server: float = 10.0

    # 5m safety
    use_quantile_margin: bool = True
    alpha: float = 0.6           # pred_safe = mean + alpha*(q90-mean)
    margin_mul: float = 0.20     # alternative: pred_safe = mean*(1+margin_mul)

    # spike detection / panic
    spike_ratio: float = 1.5     # spike if rps_true > pred_safe_5m * spike_ratio
    spike_window_k: int = 4      # lookback window K minutes
    spike_need_n: int = 3        # need N spikes in that window to confirm
    panic_burst_factor: float = 1.2

    # ddos-like gate (optional)
    enable_ddos_rule: bool = True
    ddos_top_country_min: float = 0.65
    ddos_entropy_max: float = 1.8
    ddos_dir_nunique_max: int = 25
    ddos_error_min: float = 0.05

    # anti-flap
    min_replica_floor: int = 1
    cooldown_out_min: int = 3    # block scale-in for X minutes after scale-out
    scale_in_patience_min: int = 10
    max_step_up: int = 6
    max_step_down: int = 2

    # cost / penalty (demo)
    unit_cost_per_min: float = 1.0
    penalty_per_shortage_rps_min: float = 5.0


class AutoScalerController:
    """
    Stateful controller: cooldown + low streak + spike counter window.
    Decision = max(15m, 5m, 1m panic) then apply cooldown/patience/step limits.
    """
    def __init__(self, params: PolicyParams, init_replicas: int = 1):
        self.p = params
        self.replicas = int(init_replicas)
        self.cooldown_left = 0
        self.low_streak = 0

        # rolling window of last K spike booleans for confirm
        self._spike_hist: List[bool] = []

    def _pred_safe_5m(self, pred_mean: float, pred_q90: float) -> float:
        if self.p.use_quantile_margin:
            return float(pred_mean + self.p.alpha * max(0.0, pred_q90 - pred_mean))
        return float(pred_mean * (1.0 + self.p.margin_mul))

    def _ddos_like(self, row: pd.Series) -> bool:
        if not self.p.enable_ddos_rule:
            return False
        return (
            (row["top_country_share"] >= self.p.ddos_top_country_min)
            and (row["endpoint_entropy"] <= self.p.ddos_entropy_max)
            and (int(row["dir_nunique"]) <= self.p.ddos_dir_nunique_max)
            and (row["error_rate"] >= self.p.ddos_error_min)
        )

    def step(self, row: pd.Series) -> Dict:
        """
        row requires: rps_true, pred_15m, pred_5m_mean, pred_5m_q90, plus ddos features if enabled.
        returns decision dict with replicas + event info + flags.
        """
        p = self.p
        rps_true = float(row["rps_true"])
        pred_15m = float(row["pred_15m"])
        pred_5m_mean = float(row["pred_5m_mean"])
        pred_5m_q90 = float(row["pred_5m_q90"])

        pred_safe_5m = self._pred_safe_5m(pred_5m_mean, pred_5m_q90)

        # --- 15m baseline
        rep_15m = max(p.min_replica_floor, ceil_div(pred_15m, p.cap_rps_per_server))

        # --- 5m main driver
        rep_5m = max(p.min_replica_floor, ceil_div(pred_safe_5m, p.cap_rps_per_server))

        # --- spike detection vs pred_safe_5m (NOT pred_1m)
        spike_now = bool(pred_safe_5m > 0 and rps_true > pred_safe_5m * p.spike_ratio)
        self._spike_hist.append(spike_now)
        if len(self._spike_hist) > p.spike_window_k:
            self._spike_hist = self._spike_hist[-p.spike_window_k :]
        spike_count = int(sum(self._spike_hist))
        spike_confirmed = spike_count >= p.spike_need_n

        ddos_like = self._ddos_like(row)

        # --- 1m panic: use ACTUAL load to compute needed capacity
        rep_1m = 0
        if spike_confirmed or ddos_like:
            rep_1m = max(p.min_replica_floor, ceil_div(rps_true, p.cap_rps_per_server))
            rep_1m = int(math.ceil(rep_1m * p.panic_burst_factor))

        proposed = max(rep_15m, rep_5m, rep_1m)

        # --- update low streak for scale-in patience
        # low means: actual < pred_safe_5m (or actual < capacity target) — keep it simple
        if rps_true < pred_safe_5m:
            self.low_streak += 1
        else:
            self.low_streak = 0

        old = self.replicas
        trigger = "hold"
        reason = ""

        # --- apply cooldown/patience rules
        # cooldown blocks scale-in (but allows scale-out; especially panic)
        if proposed > old:
            # scale-out fast; step limit
            new_rep = min(proposed, old + p.max_step_up)
            self.replicas = new_rep
            self.cooldown_left = p.cooldown_out_min
            trigger = "scale_out"
            reason = "panic" if (rep_1m == proposed and proposed > rep_5m) else "predictive"
        elif proposed < old:
            # scale-in only if cooldown over and low streak sufficient
            if self.cooldown_left > 0:
                trigger = "cooldown_block_scale_in"
                reason = "cooldown"
            elif self.low_streak >= p.scale_in_patience_min:
                # step-down
                new_rep = max(proposed, old - p.max_step_down)
                self.replicas = new_rep
                trigger = "scale_in"
                reason = "patience_met"
            else:
                trigger = "hold_for_patience"
                reason = f"low_streak={self.low_streak}/{p.scale_in_patience_min}"
        else:
            trigger = "hold"
            reason = "stable"

        # tick cooldown
        if self.cooldown_left > 0 and trigger != "scale_out":
            self.cooldown_left -= 1

        capacity = self.replicas * p.cap_rps_per_server
        shortage = max(0.0, rps_true - capacity)

        return {
            "timestamp": row["timestamp"],
            "rps_true": rps_true,
            "pred_15m": pred_15m,
            "pred_5m_mean": pred_5m_mean,
            "pred_5m_q90": pred_5m_q90,
            "pred_safe_5m": pred_safe_5m,
            "rep_15m": rep_15m,
            "rep_5m": rep_5m,
            "rep_1m": rep_1m,
            "replicas_final": int(self.replicas),
            "capacity_rps": float(capacity),
            "shortage_rps": float(shortage),
            "spike_now": spike_now,
            "spike_count": spike_count,
            "spike_confirmed": spike_confirmed,
            "ddos_like": ddos_like,
            "event": (trigger if int(self.replicas) != old else "none"),
            "trigger": trigger,
            "reason": reason,
            "cooldown_left": int(self.cooldown_left),
            "low_streak": int(self.low_streak),
        }


def simulate(df: pd.DataFrame, params: PolicyParams, init_replicas: int) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    ctrl = AutoScalerController(params, init_replicas=init_replicas)

    rows: List[Dict] = []
    events: List[Dict] = []

    total_cost = 0.0
    total_penalty = 0.0

    for _, row in df.iterrows():
        out = ctrl.step(row)
        rows.append(out)

        # record events when replicas changed
        if out["event"] in ("scale_out", "scale_in"):
            events.append(
                {
                    "timestamp": out["timestamp"],
                    "from": None,  # fill below for readability
                    "to": out["replicas_final"],
                    "trigger": out["trigger"],
                    "reason": out["reason"],
                    "spike_confirmed": out["spike_confirmed"],
                    "ddos_like": out["ddos_like"],
                }
            )

        # cost + penalty accumulate per minute
        total_cost += out["replicas_final"] * params.unit_cost_per_min
        total_penalty += out["shortage_rps"] * params.penalty_per_shortage_rps_min

    sim_df = pd.DataFrame(rows)

    # fill event "from" by looking at previous replica
    if events:
        ev = pd.DataFrame(events)
        # compute "from" using sim_df shift
        prev_rep = sim_df["replicas_final"].shift(1)
        rep_map = dict(zip(sim_df["timestamp"], prev_rep))
        ev["from"] = ev["timestamp"].map(rep_map).fillna(sim_df["replicas_final"].iloc[0]).astype(int)
        events_df = ev[["timestamp", "from", "to", "trigger", "reason", "spike_confirmed", "ddos_like"]].copy()
    else:
        events_df = pd.DataFrame(columns=["timestamp", "from", "to", "trigger", "reason", "spike_confirmed", "ddos_like"])

    kpi = {
        "avg_replicas": float(sim_df["replicas_final"].mean()),
        "shortage_minutes": int((sim_df["shortage_rps"] > 0).sum()),
        "total_shortage_rps_min": float(sim_df["shortage_rps"].sum()),
        "cost_total": float(total_cost),
        "penalty_total": float(total_penalty),
        "objective_cost_plus_penalty": float(total_cost + total_penalty),
    }
    return sim_df, events_df, kpi


# =========================
# Streamlit UI
# =========================


st.set_page_config(page_title="Preview 3-tier Autoscaling", layout="wide")
st.title("Preview Dashboard — Regression baseline + Autoscaling 3 tầng + Spike/DDoS + Cost (Synthetic)")

st.sidebar.header("A) Synthetic data")
minutes = st.sidebar.number_input("Minutes length", min_value=120, max_value=2880, value=720, step=60)
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=9999, value=7, step=1)
base_rps = st.sidebar.slider("Base RPS", 1.0, 50.0, 10.0, 1.0)
trend_per_hour = st.sidebar.slider("Trend per hour (RPS)", -2.0, 5.0, 0.5, 0.1)
season_amp = st.sidebar.slider("Seasonality amplitude", 0.0, 30.0, 8.0, 0.5)
noise_sigma = st.sidebar.slider("Noise sigma", 0.0, 10.0, 1.5, 0.1)

spike_prob_per_hour = st.sidebar.slider("Spike prob per hour", 0.0, 5.0, 1.2, 0.1)
spike_dur_min_lo = st.sidebar.number_input("Spike duration min (lo)", min_value=1, max_value=60, value=3, step=1)
spike_dur_min_hi = st.sidebar.number_input("Spike duration min (hi)", min_value=1, max_value=120, value=12, step=1)
spike_mag_lo = st.sidebar.slider("Spike magnitude (lo)", 1.0, 20.0, 3.0, 0.5)
spike_mag_hi = st.sidebar.slider("Spike magnitude (hi)", 1.0, 40.0, 8.0, 0.5)
ddos_prob = st.sidebar.slider("Among spikes, DDoS-like prob", 0.0, 1.0, 0.35, 0.05)

st.sidebar.header("B) Policy params (3 tiers)")
cap = st.sidebar.number_input("cap_rps_per_server", min_value=0.5, max_value=200.0, value=10.0, step=0.5)

use_quantile = st.sidebar.checkbox("Use quantile margin (mean + α*(q90-mean))", value=True)
alpha = st.sidebar.slider("α (if quantile)", 0.0, 2.0, 0.6, 0.05)
margin_mul = st.sidebar.slider("margin (if multiplicative)", 0.0, 2.0, 0.2, 0.05)

spike_ratio = st.sidebar.slider("Spike ratio threshold", 1.0, 5.0, 1.5, 0.1)
spike_window_k = st.sidebar.selectbox("Spike confirm window K (minutes)", [2, 3, 4, 5, 6], index=2)
spike_need_n = st.sidebar.selectbox("Spike need N in window", [1, 2, 3, 4, 5], index=2)
panic_burst = st.sidebar.slider("panic_burst_factor", 1.0, 3.0, 1.2, 0.05)

cooldown_out = st.sidebar.number_input("cooldown_out_min (block scale-in)", min_value=0, max_value=30, value=3, step=1)
patience = st.sidebar.number_input("scale_in_patience_min", min_value=0, max_value=60, value=10, step=1)
max_up = st.sidebar.number_input("max_step_up", min_value=1, max_value=50, value=6, step=1)
max_down = st.sidebar.number_input("max_step_down", min_value=1, max_value=50, value=2, step=1)
min_floor = st.sidebar.number_input("min_replica_floor", min_value=0, max_value=50, value=1, step=1)

st.sidebar.header("C) Cost/Penalty (demo)")
unit_cost = st.sidebar.number_input("unit_cost_per_min", min_value=0.0, max_value=100.0, value=1.0, step=0.1)
penalty_w = st.sidebar.number_input("penalty_per_shortage_rps_min", min_value=0.0, max_value=200.0, value=5.0, step=0.5)
init_rep = st.sidebar.number_input("init_replicas", min_value=0, max_value=100, value=1, step=1)

params = PolicyParams(
    cap_rps_per_server=float(cap),
    use_quantile_margin=bool(use_quantile),
    alpha=float(alpha),
    margin_mul=float(margin_mul),
    spike_ratio=float(spike_ratio),
    spike_window_k=int(spike_window_k),
    spike_need_n=int(spike_need_n),
    panic_burst_factor=float(panic_burst),
    min_replica_floor=int(min_floor),
    cooldown_out_min=int(cooldown_out),
    scale_in_patience_min=int(patience),
    max_step_up=int(max_up),
    max_step_down=int(max_down),
    unit_cost_per_min=float(unit_cost),
    penalty_per_shortage_rps_min=float(penalty_w),
)

st.sidebar.divider()
run_btn = st.sidebar.button("Generate + Run simulation", type="primary")


def render_dashboard(sim_df: pd.DataFrame, events_df: pd.DataFrame, kpi: Dict, title_suffix: str = ""):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Avg replicas", f"{kpi['avg_replicas']:.2f}")
    c2.metric("Shortage minutes", f"{kpi['shortage_minutes']}")
    c3.metric("Total shortage (RPS-min)", f"{kpi['total_shortage_rps_min']:.2f}")
    c4.metric("Cost total", f"{kpi['cost_total']:.2f}")
    c5.metric("Objective (cost+penalty)", f"{kpi['objective_cost_plus_penalty']:.2f}")

    st.subheader(f"Demand vs Forecast (5m safe, 15m baseline){title_suffix}")
    plot1 = sim_df[["timestamp", "rps_true", "pred_safe_5m", "pred_15m"]].melt(
        id_vars=["timestamp"], var_name="series", value_name="rps"
    )
    fig1 = px.line(plot1, x="timestamp", y="rps", color="series")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader(f"Replicas per tier + final{title_suffix}")
    plot2 = sim_df[["timestamp", "rep_15m", "rep_5m", "rep_1m", "replicas_final"]].melt(
        id_vars=["timestamp"], var_name="series", value_name="replicas"
    )
    fig2 = px.line(plot2, x="timestamp", y="replicas", color="series")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader(f"Capacity vs Demand (RPS){title_suffix}")
    plot3 = sim_df[["timestamp", "rps_true", "capacity_rps"]].melt(
        id_vars=["timestamp"], var_name="series", value_name="rps"
    )
    fig3 = px.line(plot3, x="timestamp", y="rps", color="series")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Scale events log")
    st.dataframe(events_df, use_container_width=True)

    with st.expander("Show simulation dataframe (head)"):
        st.dataframe(sim_df.head(50), use_container_width=True)


# Session state for playback
if "sim_df" not in st.session_state:
    st.session_state.sim_df = None
if "events_df" not in st.session_state:
    st.session_state.events_df = None
if "kpi" not in st.session_state:
    st.session_state.kpi = None
if "play" not in st.session_state:
    st.session_state.play = False
if "idx" not in st.session_state:
    st.session_state.idx = 0

if run_btn:
    df = generate_synthetic_1m(
        minutes=int(minutes),
        seed=int(seed),
        base_rps=float(base_rps),
        trend_per_hour=float(trend_per_hour),
        season_amp=float(season_amp),
        noise_sigma=float(noise_sigma),
        spike_prob_per_hour=float(spike_prob_per_hour),
        spike_duration_min_range=(int(spike_dur_min_lo), int(spike_dur_min_hi)),
        spike_magnitude_range=(float(spike_mag_lo), float(spike_mag_hi)),
        ddos_prob=float(ddos_prob),
    )
    df = add_baseline_forecasts(df)

    sim_df, events_df, kpi = simulate(df, params=params, init_replicas=int(init_rep))

    st.session_state.sim_df = sim_df
    st.session_state.events_df = events_df
    st.session_state.kpi = kpi
    st.session_state.play = False
    st.session_state.idx = 0

if st.session_state.sim_df is None:
    st.info("Bấm **Generate + Run simulation** để tạo dữ liệu và xem dashboard.")
    st.stop()

# Tabs: Full vs Playback
tab1, tab2 = st.tabs(["Full view", "Playback (fake realtime)"])

with tab1:
    render_dashboard(st.session_state.sim_df, st.session_state.events_df, st.session_state.kpi)

with tab2:
    st.caption("Playback = frontend phát lại dần theo thời gian (giả lập realtime).")

    speed = st.slider("Speed (sec/step)", 0.05, 1.0, 0.20, 0.05)
    step_size = st.selectbox("Step size (points per tick)", [1, 2, 5, 10], index=2)

    b1, b2, b3, b4 = st.columns(4)
    if b1.button("Play"):
        st.session_state.play = True
    if b2.button("Pause"):
        st.session_state.play = False
    if b3.button("Step"):
        st.session_state.idx = min(len(st.session_state.sim_df), st.session_state.idx + step_size)
    if b4.button("Reset"):
        st.session_state.play = False
        st.session_state.idx = 0

    # render placeholders
    chart_a = st.empty()
    chart_b = st.empty()
    chart_c = st.empty()
    ev_tbl = st.empty()

    # auto-advance loop (simple blocking loop; ok for demo)
    if st.session_state.play:
        # advance a few ticks then stop to avoid locking UI too long
        ticks = 25
        for _ in range(ticks):
            st.session_state.idx = min(len(st.session_state.sim_df), st.session_state.idx + step_size)
            df_part = st.session_state.sim_df.iloc[: st.session_state.idx].copy()

            p1 = df_part[["timestamp", "rps_true", "pred_safe_5m", "pred_15m"]].melt(
                id_vars=["timestamp"], var_name="series", value_name="rps"
            )
            chart_a.plotly_chart(px.line(p1, x="timestamp", y="rps", color="series"), use_container_width=True)

            p2 = df_part[["timestamp", "rep_15m", "rep_5m", "rep_1m", "replicas_final"]].melt(
                id_vars=["timestamp"], var_name="series", value_name="replicas"
            )
            chart_b.plotly_chart(px.line(p2, x="timestamp", y="replicas", color="series"), use_container_width=True)

            p3 = df_part[["timestamp", "rps_true", "capacity_rps"]].melt(
                id_vars=["timestamp"], var_name="series", value_name="rps"
            )
            chart_c.plotly_chart(px.line(p3, x="timestamp", y="rps", color="series"), use_container_width=True)

            cur_ts = df_part["timestamp"].iloc[-1]
            ev_part = st.session_state.events_df[pd.to_datetime(st.session_state.events_df["timestamp"]) <= pd.to_datetime(cur_ts)]
            ev_tbl.dataframe(ev_part.tail(20), use_container_width=True)

            if st.session_state.idx >= len(st.session_state.sim_df):
                st.session_state.play = False
                break

            time.sleep(speed)

        # force a rerun so UI stays responsive during play
        st.experimental_rerun()
    else:
        # paused view at current idx
        df_part = st.session_state.sim_df.iloc[: max(1, st.session_state.idx)].copy()

        p1 = df_part[["timestamp", "rps_true", "pred_safe_5m", "pred_15m"]].melt(
            id_vars=["timestamp"], var_name="series", value_name="rps"
        )
        chart_a.plotly_chart(px.line(p1, x="timestamp", y="rps", color="series"), use_container_width=True)

        p2 = df_part[["timestamp", "rep_15m", "rep_5m", "rep_1m", "replicas_final"]].melt(
            id_vars=["timestamp"], var_name="series", value_name="replicas"
        )
        chart_b.plotly_chart(px.line(p2, x="timestamp", y="replicas", color="series"), use_container_width=True)

        p3 = df_part[["timestamp", "rps_true", "capacity_rps"]].melt(
            id_vars=["timestamp"], var_name="series", value_name="rps"
        )
        chart_c.plotly_chart(px.line(p3, x="timestamp", y="rps", color="series"), use_container_width=True)

        cur_ts = df_part["timestamp"].iloc[-1]
        ev_part = st.session_state.events_df[pd.to_datetime(st.session_state.events_df["timestamp"]) <= pd.to_datetime(cur_ts)]
        ev_tbl.dataframe(ev_part.tail(20), use_container_width=True)

    st.caption(f"Playback index: {st.session_state.idx}/{len(st.session_state.sim_df)}")
