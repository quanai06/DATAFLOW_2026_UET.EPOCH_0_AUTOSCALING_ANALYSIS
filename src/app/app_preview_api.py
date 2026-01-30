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

with st.sidebar:
    st.header("Backend")
    base_url = st.text_input("API base_url", value=os.getenv("AUTOSCALER_API", "http://localhost:8000"))

    st.divider()
    st.header("Chọn chiến lược")
    model = st.text_input("model", value="xgboost")
    strategy = st.selectbox("strategy", ["reactive", "predictive", "hybrid"], index=2)

    st.divider()
    st.header("Nguồn dữ liệu")
    use_server_side_csv = st.checkbox(
        "API tự đọc merged CSV từ server (simulate-best)",
        value=True,
        help="Nếu backend của bạn có sẵn file results/merged_<model>_data.csv. Nếu chạy local thì để True."
    )
    merged_csv_path = st.text_input(
        "merged_csv_path (optional)",
        value="",
        help="Bỏ trống để backend dùng mặc định results/merged_<model>_data.csv"
    )

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
            resp = _api_post_multipart(base_url, "/simulate", data=data, file_bytes=file_bytes, filename=uploaded.name)

        # unpack
        kpi = resp["kpi"]
        sim = pd.DataFrame(resp["sim"])
        events = pd.DataFrame(resp["events"])

        # metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Avg servers", f"{kpi['avg_servers']:.2f}")
        m2.metric("Max servers", f"{kpi['max_servers']}")
        m3.metric("Scaling events", f"{kpi['scaling_events']}")
        m4.metric("Total score", f"{kpi['total_score']:.2f}")

        st.divider()

        # charts
        if "timestamp" in sim.columns:
            sim["timestamp"] = pd.to_datetime(sim["timestamp"], errors="coerce")

        st.subheader("Servers over time")
        fig1 = px.line(sim, x="timestamp", y="servers")
        st.plotly_chart(fig1, use_container_width=True)

        # Demand vs capacity (req)
        req_cols = [c for c in ["act_1m_req", "cap_req_eff", "cap_req_hard"] if c in sim.columns]
        if len(req_cols) >= 2:
            st.subheader("Request: Actual vs Capacity")
            dfp = sim[["timestamp"] + req_cols].melt(id_vars=["timestamp"], var_name="series", value_name="value")
            st.plotly_chart(px.line(dfp, x="timestamp", y="value", color="series"), use_container_width=True)

        # Demand vs capacity (bytes)
        bytes_cols = [c for c in ["act_1m_bytes", "cap_bytes_eff", "cap_bytes_hard"] if c in sim.columns]
        if len(bytes_cols) >= 2:
            st.subheader("Bytes: Actual vs Capacity")
            dfp = sim[["timestamp"] + bytes_cols].melt(id_vars=["timestamp"], var_name="series", value_name="value")
            st.plotly_chart(px.line(dfp, x="timestamp", y="value", color="series"), use_container_width=True)

        # overloads
        overload_cols = [c for c in ["soft_overload_req", "hard_overload_req", "soft_overload_bytes", "hard_overload_bytes"] if c in sim.columns]
        if overload_cols:
            st.subheader("Overload (soft/hard)")
            dfp = sim[["timestamp"] + overload_cols].melt(id_vars=["timestamp"], var_name="series", value_name="value")
            st.plotly_chart(px.line(dfp, x="timestamp", y="value", color="series"), use_container_width=True)

        st.subheader("Scale events")
        st.dataframe(events, use_container_width=True)

        with st.expander("Show sim dataframe"):
            st.dataframe(sim.head(200), use_container_width=True)

    except Exception as e:
        st.error(f"Run failed: {e}")
