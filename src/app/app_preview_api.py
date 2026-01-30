# app_preview_api.py
# Streamlit dashboard that calls the FastAPI backend (api_scaler.py)
#
# Run:
#   1) uvicorn api_scaler:app --reload --port 8000
#   2) streamlit run app_preview_api.py

import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import time


# =========================
# Config
# =========================
st.set_page_config(page_title="Autoscaling Dashboard (API)", layout="wide")


def _api_get(base_url: str, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    r = requests.get(base_url.rstrip("/") + path, params=params, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"GET {path} failed: {r.status_code} - {r.text}")
    return r.json()


def _api_post_json(base_url: str, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(base_url.rstrip("/") + path, json=payload, timeout=300)
    if r.status_code != 200:
        raise RuntimeError(f"POST {path} failed: {r.status_code} - {r.text}")
    return r.json()


def _api_post_multipart(
    base_url: str,
    path: str,
    data: Dict[str, Any],
    file_bytes: bytes,
    filename: str = "merged.csv",
) -> Dict[str, Any]:
    files = {"file": (filename, file_bytes, "text/csv")}
    r = requests.post(base_url.rstrip("/") + path, data=data, files=files, timeout=300)
    if r.status_code != 200:
        raise RuntimeError(f"POST {path} failed: {r.status_code} - {r.text}")
    return r.json()


# =========================
# UI
# =========================
st.title("Autoscaling Dashboard — gọi backend FastAPI")

if "last_resp" not in st.session_state:
    st.session_state.last_resp = None
if "sim" not in st.session_state:
    st.session_state.sim = None
if "events" not in st.session_state:
    st.session_state.events = None
if "kpi" not in st.session_state:
    st.session_state.kpi = None
# playback state
if "play" not in st.session_state:
    st.session_state.play = False
if "idx" not in st.session_state:
    st.session_state.idx = 0

with st.sidebar:
    st.header("Backend")
    base_url = st.text_input("API base_url", value=os.getenv("AUTOSCALER_API", "http://localhost:8000"))

    
    st.divider()
    st.header("Chọn chiến lược")
    model = st.selectbox("model",["xgboost","lightgbm","lstm"],index =2)
    strategy = st.selectbox("strategy", ["reactive", "predictive", "hybrid"], index=2)

    st.divider()
    st.header("Nguồn dữ liệu")
    use_server_side_csv = st.checkbox(
        "API tự đọc merged CSV từ server ",
        value=True,
        help="Nếu backend của bạn có sẵn file results/merged_<model>_data.csv. Nếu chạy local thì để True."
    )
    merged_csv_path = st.text_input(
        "merged_csv_path (optional)",
        value="",
        help="Bỏ trống để backend dùng mặc định results/merged_<model>_data.csv"
    )
    params = {"model": model}
    if merged_csv_path and merged_csv_path.strip():
        params["merged_csv_path"] = merged_csv_path.strip()

    start_time, end_time = "", ""

    try:
        r = requests.get(base_url.rstrip("/") + "/time-range", params=params, timeout=60)
        if r.ok and r.json().get("ok"):
            min_ts = pd.to_datetime(r.json()["min_timestamp"])
            max_ts = pd.to_datetime(r.json()["max_timestamp"])

            st.divider()
            st.header("Khoảng thời gian (kéo chọn)")
            range_vals = st.slider(
                "Chọn khoảng",
                min_value=min_ts.to_pydatetime(),
                max_value=max_ts.to_pydatetime(),
                value=(min_ts.to_pydatetime(), max_ts.to_pydatetime()),
                format="YYYY-MM-DD HH:mm:ss",
            )
            start_time = range_vals[0].strftime("%Y-%m-%d %H:%M:%S")
            end_time = range_vals[1].strftime("%Y-%m-%d %H:%M:%S")
        else:
            st.warning("Không lấy được time range từ backend, fallback sang nhập tay.")
            start_time = st.text_input("start_time (YYYY-MM-DD HH:MM:SS)", value="")
            end_time = st.text_input("end_time (YYYY-MM-DD HH:MM:SS)", value="")
    except Exception as _:
        st.warning("Không gọi được /time-range, fallback sang nhập tay.")
        start_time = st.text_input("start_time (YYYY-MM-DD HH:MM:SS)", value="")
        end_time = st.text_input("end_time (YYYY-MM-DD HH:MM:SS)", value="")

    uploaded = None
    if not use_server_side_csv:
        uploaded = st.file_uploader("Upload merged_..._data.csv", type=["csv"])

    st.divider()
    st.header("Init & run")
    init_servers = st.number_input("init_servers", min_value=1, max_value=10000, value=1, step=1)
    run_btn = st.button("Run simulation", type="primary")


# =========================
# Health check
# =========================
try:
    health = _api_get(base_url, "/health")
    st.caption(f"API OK — cap_req={health['cap_req']}, cap_bytes={health['cap_bytes']}, target_util={health['target_util']}, max_servers={health['max_servers']}")
except Exception as e:
    st.error(f"Không kết nối được API: {e}")
    st.stop()


# =========================
# Fetch best params + render
# =========================
colA, colB = st.columns([1, 1])
with colA:
    st.subheader("Best params (Optuna)")
    try:
        bp = _api_get(base_url, "/best-params", params={"model": model, "strategy": strategy})
        st.json(bp)
        normalized_params = bp.get("normalized_params", {})
    except Exception as e:
        st.warning(f"Chưa load được best params: {e}")
        normalized_params = {}

with colB:
    st.subheader("Ghi chú")
    st.markdown(
        """
- `/best-params` trả cả **raw_best_params** (tên param theo Optuna: k_base_h, ...) và **normalized_params** (tên param đúng để chạy UniversalOptimizer).
- Dashboard sẽ chạy bằng **normalized_params**.
        """
    )


# =========================
# Run simulation
# =========================
if run_btn:
    try:
        if use_server_side_csv:
            payload = {
                "model": model,
                "strategy": strategy,
                "merged_csv_path": merged_csv_path.strip() or None,
                "init_servers": int(init_servers),
            }
            if start_time.strip():
                payload["start_time"] = start_time.strip()
            if end_time.strip():
                payload["end_time"] = end_time.strip()

            resp = _api_post_json(base_url, "/simulate-best", payload)

        else:
            if uploaded is None:
                st.error("Bạn phải upload file merged CSV khi bỏ chọn simulate-best.")
                st.stop()

            file_bytes = uploaded.getvalue()
            data = {
                "strategy": strategy,
                "params_json": json.dumps(normalized_params, ensure_ascii=False),
                "init_servers": str(int(init_servers)),
            }
            # gửi time-range luôn cho nhánh upload
            if start_time.strip():
                data["start_time"] = start_time.strip()
            if end_time.strip():
                data["end_time"] = end_time.strip()

            resp = _api_post_multipart(base_url, "/simulate", data=data, file_bytes=file_bytes, filename=uploaded.name)

        # unpack
        kpi = resp["kpi"]
        sim = pd.DataFrame(resp["sim"])
        events = pd.DataFrame(resp["events"])

        # parse + sort timestamp
        if "timestamp" in sim.columns:
            sim["timestamp"] = pd.to_datetime(sim["timestamp"], errors="coerce")
            sim = sim.sort_values("timestamp").reset_index(drop=True)
        if "timestamp" in events.columns:
            events["timestamp"] = pd.to_datetime(events["timestamp"], errors="coerce")
            events = events.sort_values("timestamp").reset_index(drop=True)

        # save to session_state
        st.session_state.kpi = kpi
        st.session_state.sim = sim
        st.session_state.events = events
        st.session_state.last_resp = resp

        # reset playback index mỗi lần run để khỏi dính idx cũ
        st.session_state.play = False
        st.session_state.idx = 0

        st.success("Simulation loaded ✓")

    except Exception as e:
        st.error(f"Run failed: {e}")
if st.session_state.sim is None:
    st.info("Nhấn **Run simulation** để chạy mô phỏng.")
    st.stop()

# lấy từ state
kpi = st.session_state.kpi
sim = st.session_state.sim
events = st.session_state.events

# guard idx
if st.session_state.idx > len(sim):
    st.session_state.idx = 0

# metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Avg servers", f"{kpi['avg_servers']:.2f}")
m2.metric("Max servers", f"{kpi['max_servers']}")
m3.metric("Scaling events", f"{kpi['scaling_events']}")
m4.metric("Total Cost", f"{kpi['total_score']:.2f}")

st.divider()

tab_full, tab_play = st.tabs(["Full view", "Playback"])

def _render_charts(sim_df: pd.DataFrame):
    st.subheader("Servers over time")
    st.plotly_chart(px.line(sim_df, x="timestamp", y="servers"), use_container_width=True)

    req_cols = [c for c in ["act_1m_req", "cap_req_eff", "cap_req_hard"] if c in sim_df.columns]
    if len(req_cols) >= 2:
        st.subheader("Request: Actual vs Capacity")
        dfp = sim_df[["timestamp"] + req_cols].melt(id_vars=["timestamp"], var_name="series", value_name="value")
        st.plotly_chart(px.line(dfp, x="timestamp", y="value", color="series"), use_container_width=True)

    bytes_cols = [c for c in ["act_1m_bytes", "cap_bytes_eff", "cap_bytes_hard"] if c in sim_df.columns]
    if len(bytes_cols) >= 2:
        st.subheader("Bytes: Actual vs Capacity")
        dfp = sim_df[["timestamp"] + bytes_cols].melt(id_vars=["timestamp"], var_name="series", value_name="value")
        st.plotly_chart(px.line(dfp, x="timestamp", y="value", color="series"), use_container_width=True)

    overload_cols = [c for c in ["soft_overload_req", "hard_overload_req", "soft_overload_bytes", "hard_overload_bytes"] if c in sim_df.columns]
    if overload_cols:
        st.subheader("Overload (soft/hard)")
        dfp = sim_df[["timestamp"] + overload_cols].melt(id_vars=["timestamp"], var_name="series", value_name="value")
        st.plotly_chart(px.line(dfp, x="timestamp", y="value", color="series"), use_container_width=True)

    spike_cols = [c for c in ["spike_now", "spike_confirmed", "spike_now_req", "spike_now_bytes",
                              "spike_confirmed_req", "spike_confirmed_bytes"] if c in sim_df.columns]
    if spike_cols:
        st.subheader("Spike signals")
        st.dataframe(sim_df[["timestamp"] + spike_cols], use_container_width=True)

with tab_full:
    _render_charts(sim)

    st.subheader("Scale events")
    events_display = events[["timestamp", "from", "to","spike_confirmed"]]
    st.dataframe(events_display, use_container_width=True)

    with st.expander("Show sim dataframe"):
        st.dataframe(sim, use_container_width=True)

with tab_play:
    speed = st.slider("Speed (sec/step)", 0.05, 1.0, 0.20, 0.05)
    step_size = st.selectbox("Step size (points per tick)", [1, 2, 5, 10], index=2)

    b1, b2, b3, b4 = st.columns(4)
    if b1.button("Play"):
        st.session_state.play = True
    if b2.button("Pause"):
        st.session_state.play = False
    if b3.button("Step"):
        st.session_state.idx = min(len(sim), st.session_state.idx + step_size)
    if b4.button("Reset"):
        st.session_state.play = False
        st.session_state.idx = 0

    chart_area = st.empty()
    ev_area = st.empty()

    def _render_play(idx: int):
        part = sim.iloc[:max(1, idx)].copy()
        with chart_area.container():
            _render_charts(part)

        if (not events.empty) and ("timestamp" in events.columns) and (part["timestamp"].notna().any()):
            cur_ts = part["timestamp"].dropna().iloc[-1]
            ev_part = events[events["timestamp"] <= cur_ts]
        else:
            ev_part = events

        with ev_area.container():
            st.subheader("Scale events")
            ev = ev_part[["timestamp", "from", "to","spike_confirmed"]]
            st.dataframe(ev, use_container_width=True)

    if st.session_state.play:
        ticks = 25  # giới hạn số tick mỗi rerun
        for _ in range(ticks):
            st.session_state.idx = min(len(sim), st.session_state.idx + step_size)
            _render_play(st.session_state.idx)

            if st.session_state.idx >= len(sim):
                st.session_state.play = False
                break

            time.sleep(speed)

        # rerun để tiếp tục playback
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()
    else:
        _render_play(st.session_state.idx)

