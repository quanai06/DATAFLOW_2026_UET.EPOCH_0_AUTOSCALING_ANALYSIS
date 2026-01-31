"""FastAPI backend for autoscaling simulator & recommender.

This wraps your optimizer + Optuna-best-params JSON into an HTTP API so Streamlit
can call it.

Run:
  pip install fastapi uvicorn python-dotenv pandas python-multipart
  uvicorn api_scaler:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import io
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field


from optimize.universalOptimizer import UniversalOptimizer


# ============================================================
# Config (same env keys as runOptimizer.py)
# ============================================================
load_dotenv()


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    return float(v) if v not in (None, "") else float(default)


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    return int(v) if v not in (None, "") else int(default)


CAP_REQ = _env_float("CAP_REQ", 100.0)
CAP_BYTES = _env_float("CAP_BYTES", 1e6)
TARGET_UTIL = _env_float("TARGET_UTIL", 0.7)
MAX_SERVERS = _env_int("MAX_SERVERS", 200)

COST_PER_SERVER_PER_MIN = _env_int("COST_PER_SERVER_PER_MIN", 1)
COST_SOFT_SLA_PENALTY_REQ = _env_int("COST_SOFT_SLA_PENALTY_REQ", 1)
COST_SOFT_SLA_PENALTY_BYTES = _env_int("COST_SOFT_SLA_PENALTY_BYTES", 1)
COST_HARD_SLA_PENALTY_REQ = _env_int("COST_HARD_SLA_PENALTY_REQ", 20)
COST_HARD_SLA_PENALTY_BYTES = _env_int("COST_HARD_SLA_PENALTY_BYTES", 20)
COST_SCALING_EVENT = _env_int("COST_SCALING_EVENT", 1)

EFF_CAP_REQ = CAP_REQ * TARGET_UTIL
EFF_CAP_BYTES = CAP_BYTES * TARGET_UTIL


# ============================================================
# Param normalization (Optuna stores trial names, not your internal keys)
# ============================================================

def normalize_best_params(strategy: str, best_params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Optuna trial param names -> UniversalOptimizer expected param keys."""
    s = strategy.lower()

    if s == "reactive":
        # study.best_params keys: max_reduce_r, in_patience_r, out_patience_r
        return {
            "max_reduce": int(best_params["max_reduce_r"]),
            "in_patience": int(best_params["in_patience_r"]),
            "out_patience": int(best_params["out_patience_r"]),
        }

    if s == "predictive":
        # keys: k_base_p, alpha_5m_p, in_patience_p
        return {
            "k_base": float(best_params["k_base_p"]),
            "alpha_5m": float(best_params["alpha_5m_p"]),
            "in_patience": int(best_params["in_patience_p"]),
        }

    if s == "hybrid":
        # keys: k_base_h, alpha_5m_h, panic_ratio_h, burst_add_h, in_patience_h, max_reduce_h
        return {
            "k_base": float(best_params["k_base_h"]),
            "alpha_5m": float(best_params["alpha_5m_h"]),
            "panic_ratio": float(best_params["panic_ratio_h"]),
            "burst_add": int(best_params["burst_add_h"]),
            "in_patience": int(best_params["in_patience_h"]),
            "max_reduce": int(best_params["max_reduce_h"]),
        }

    raise ValueError(f"Unknown strategy: {strategy}")


def load_best_params_file(model: str, strategy: str, base_dir: str = "results/optimize") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Returns (raw_best, normalized_best)."""
    path = os.path.join(base_dir, model, f"{model}_{strategy}_best_strategy_params.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Best params file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    raw_best = payload.get("best_params") or {}
    normalized = normalize_best_params(strategy, raw_best)
    return raw_best, normalized


# ============================================================
# Simulation core (matches runOptimizer.objective accounting)
# ============================================================

def simulate_df(
    df: pd.DataFrame,
    strategy: str,
    params: Dict[str, Any],
    init_servers: int = 1,
) -> Dict[str, Any]:
    """Run UniversalOptimizer over df rows and return results for dashboard."""
    opt = UniversalOptimizer(
        mode=strategy,
        cap_req=CAP_REQ,
        cap_bytes=CAP_BYTES,
        target_util=TARGET_UTIL,
        max_servers=MAX_SERVERS,
        **params,
    )

    # force initial servers if you want deterministic start
    try:
        opt.current_servers = max(1, int(init_servers))
    except Exception:
        pass

    replicas: List[int] = []
    scaling_events = 0
    prev_s = int(opt.current_servers)

    total_soft_overload_req = 0.0
    total_soft_overload_bytes = 0.0
    total_hard_overload_req = 0.0
    total_hard_overload_bytes = 0.0

# =========================
    # Spike detection config
    # =========================
    spike_ratio = float(params.get("spike_ratio", 1.5))  # act > safe * ratio => spike_now
    spike_win_k = int(params.get("spike_win_k", 5))      # window K phút
    spike_win_n = int(params.get("spike_win_n", 3))      # >=N lần spike trong K => confirmed

    _hist_req: List[bool] = []
    _hist_bytes: List[bool] = []

    spike_now_req_list: List[bool] = []
    spike_now_bytes_list: List[bool] = []
    spike_conf_req_list: List[bool] = []
    spike_conf_bytes_list: List[bool] = []
    spike_now_list: List[bool] = []
    spike_conf_list: List[bool] = []
    # loop
    for t, row in enumerate(df.itertuples(index=False)):
        s = int(opt.step(row, t))
        replicas.append(s)

        # penalties are per-minute "area" in (req/s)*min and (bytes/s)*min
        # soft overload counts only above effective cap but below hard cap
        act_req = float(getattr(row, "act_1m_req"))
        act_bytes = float(getattr(row, "act_1m_bytes"))

         # =========================
        # Spike detection (req & bytes) vs safe_5m
        # =========================
        alpha = float(params.get("alpha_5m", 0.0))

        # ---- bytes safe_5m
        spike_now_bytes = False
        try:
            f5_b = float(getattr(row, "predicted_5m_bytes"))
            f5q_b = float(getattr(row, "predicted_5m_q90_bytes"))
            safe_b = f5_b + alpha * max(0.0, (f5q_b - f5_b))
            if safe_b > 0:
                spike_now_bytes = (act_bytes > safe_b * spike_ratio)
        except Exception:
            # fallback: coi spike nếu vượt hard cap hiện tại
            spike_now_bytes = (act_bytes > s * CAP_BYTES)

        # ---- req safe_5m
        spike_now_req = False
        try:
            f5_r = float(getattr(row, "predicted_5m_req"))
            f5q_r = float(getattr(row, "predicted_5m_q90_req"))
            safe_r = f5_r + alpha * max(0.0, (f5q_r - f5_r))
            if safe_r > 0:
                spike_now_req = (act_req > safe_r * spike_ratio)
        except Exception:
            spike_now_req = (act_req > s * CAP_REQ)

        # confirmed logic
        _hist_bytes.append(bool(spike_now_bytes))
        if len(_hist_bytes) > spike_win_k:
            _hist_bytes = _hist_bytes[-spike_win_k:]
        spike_conf_bytes = (sum(_hist_bytes) >= spike_win_n)

        _hist_req.append(bool(spike_now_req))
        if len(_hist_req) > spike_win_k:
            _hist_req = _hist_req[-spike_win_k:]
        spike_conf_req = (sum(_hist_req) >= spike_win_n)

        spike_now = bool(spike_now_req or spike_now_bytes)
        spike_conf = bool(spike_conf_req or spike_conf_bytes)

        spike_now_req_list.append(bool(spike_now_req))
        spike_now_bytes_list.append(bool(spike_now_bytes))
        spike_conf_req_list.append(bool(spike_conf_req))
        spike_conf_bytes_list.append(bool(spike_conf_bytes))
        spike_now_list.append(spike_now)
        spike_conf_list.append(spike_conf)

        soft_overload_req = max(0.0, min(act_req, s * CAP_REQ) - s * EFF_CAP_REQ)
        soft_overload_bytes = max(0.0, min(act_bytes, s * CAP_BYTES) - s * EFF_CAP_BYTES)

        hard_overload_req = max(0.0, act_req - s * CAP_REQ)
        hard_overload_bytes = max(0.0, act_bytes - s * CAP_BYTES)

        total_soft_overload_req += soft_overload_req
        total_soft_overload_bytes += soft_overload_bytes
        total_hard_overload_req += hard_overload_req
        total_hard_overload_bytes += hard_overload_bytes

        if s != prev_s:
            scaling_events += 1
        prev_s = s

    # cost components
    count_servers = float(sum(replicas))
    sla_penalty = (
        total_soft_overload_req * COST_SOFT_SLA_PENALTY_REQ
        + total_soft_overload_bytes * COST_SOFT_SLA_PENALTY_BYTES
        + total_hard_overload_req * COST_HARD_SLA_PENALTY_REQ
        + total_hard_overload_bytes * COST_HARD_SLA_PENALTY_BYTES
    )
    total_score = count_servers * COST_PER_SERVER_PER_MIN + sla_penalty + scaling_events * COST_SCALING_EVENT

    # build outputs
    out = df.copy()
    out["servers"] = replicas

    out["spike_now_req"] = spike_now_req_list
    out["spike_now_bytes"] = spike_now_bytes_list
    out["spike_confirmed_req"] = spike_conf_req_list
    out["spike_confirmed_bytes"] = spike_conf_bytes_list
    out["spike_now"] = spike_now_list
    out["spike_confirmed"] = spike_conf_list

    # helpful columns for charting
    out["cap_req_eff"] = out["servers"] * EFF_CAP_REQ
    out["cap_bytes_eff"] = out["servers"] * EFF_CAP_BYTES
    out["cap_req_hard"] = out["servers"] * CAP_REQ
    out["cap_bytes_hard"] = out["servers"] * CAP_BYTES

    out["soft_overload_req"] = (out[["act_1m_req"]].clip(upper=out["cap_req_hard"], axis=0)["act_1m_req"] - out["cap_req_eff"]).clip(lower=0)
    out["soft_overload_bytes"] = (out[["act_1m_bytes"]].clip(upper=out["cap_bytes_hard"], axis=0)["act_1m_bytes"] - out["cap_bytes_eff"]).clip(lower=0)
    out["hard_overload_req"] = (out["act_1m_req"] - out["cap_req_hard"]).clip(lower=0)
    out["hard_overload_bytes"] = (out["act_1m_bytes"] - out["cap_bytes_hard"]).clip(lower=0)

    # events table
    out["prev_servers"] = out["servers"].shift(1).fillna(out["servers"].iloc[0]).astype(int)
    event_cols = ["timestamp", "prev_servers", "servers",
                  "spike_now", "spike_confirmed",
                  "spike_now_req", "spike_now_bytes",
                  "spike_confirmed_req", "spike_confirmed_bytes"]

    # chỉ lấy các cột nào thực sự tồn tại (an toàn)
    event_cols = [c for c in event_cols if c in out.columns]

    events = out.loc[(out["servers"] != out["prev_servers"]) | out["spike_confirmed"]== True, event_cols].copy()
    events = events.rename(columns={"prev_servers": "from", "servers": "to"}).reset_index(drop=True)

    kpi = {
        "avg_servers": float(out["servers"].mean()),
        "max_servers": int(out["servers"].max()),
        "scaling_events": int(scaling_events),
        "total_soft_overload": float(total_soft_overload_req + total_soft_overload_bytes),
        "total_hard_overload": float(total_hard_overload_req + total_hard_overload_bytes),
        "server_minutes": float(count_servers),
        "sla_penalty": float(sla_penalty),
        "total_score": float(total_score),
    }

    return {
        "kpi": kpi,
        "events": events.to_dict(orient="records"),
        "sim": out.to_dict(orient="records"),
    }


# ============================================================
# API schemas
# ============================================================

class BestParamsResponse(BaseModel):
    model: str
    strategy: str
    raw_best_params: Dict[str, Any]
    normalized_params: Dict[str, Any]


class SimulateBestRequest(BaseModel):
    model: str = Field(..., description="e.g., xgboost")
    strategy: str = Field(..., description="reactive | predictive | hybrid")
    merged_csv_path: Optional[str] = Field(
        None,
        description="If omitted, defaults to results/merged_<model>_data.csv",
    )
    init_servers: int = 1
    start_time: Optional[str] = Field(None, description="ISO datetime, e.g. 1995-08-23 00:00:00")
    end_time: Optional[str] = Field(None, description="ISO datetime, e.g. 1995-08-31 23:59:59")


class SimulateRequestJSON(BaseModel):
    strategy: str
    params: Dict[str, Any]
    rows: List[Dict[str, Any]]
    init_servers: int = 1


# ============================================================
# FastAPI app
# ============================================================

app = FastAPI(title="Autoscaling Optimizer API", version="1.0")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "cap_req": CAP_REQ,
        "cap_bytes": CAP_BYTES,
        "target_util": TARGET_UTIL,
        "max_servers": MAX_SERVERS,
    }


@app.get("/best-params", response_model=BestParamsResponse)
def best_params(model: str, strategy: str) -> BestParamsResponse:
    try:
        raw_best, normalized = load_best_params_file(model=model, strategy=strategy)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return BestParamsResponse(
        model=model,
        strategy=strategy,
        raw_best_params=raw_best,
        normalized_params=normalized,
    )
@app.get("/time-range")
def time_range(model: str, merged_csv_path: Optional[str] = None) -> Dict[str, Any]:
    csv_path = merged_csv_path or f"results/merged_{model}_data.csv"
    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns:
        return {"ok": False, "error": "No timestamp column", "csv_path": csv_path}

    ts = pd.to_datetime(df["timestamp"], errors="coerce").dropna()
    if ts.empty:
        return {"ok": False, "error": "No valid timestamps", "csv_path": csv_path}

    return {
        "ok": True,
        "csv_path": csv_path,
        "min_timestamp": ts.min().isoformat(sep=" "),
        "max_timestamp": ts.max().isoformat(sep=" "),
    }


@app.post("/simulate-best")
def simulate_best(req: SimulateBestRequest) -> Dict[str, Any]:
    try:
        _, params = load_best_params_file(model=req.model, strategy=req.strategy)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    csv_path = req.merged_csv_path or f"results/merged_{req.model}_data.csv"
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail=f"Merged CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        if req.start_time:
            start = pd.to_datetime(req.start_time, errors="coerce")
            if pd.notna(start):
                df = df[df["timestamp"] >= start]
        if req.end_time:
            end = pd.to_datetime(req.end_time, errors="coerce")
            if pd.notna(end):
                df = df[df["timestamp"] <= end]

        df = df.reset_index(drop=True)

    return simulate_df(df=df, strategy=req.strategy, params=params, init_servers=req.init_servers)


@app.post("/simulate")
def simulate(
    strategy: str,
    params_json: str,
    init_servers: int = 1,
    file: UploadFile = File(...),
) -> Dict[str, Any]:
    """Upload a merged CSV and simulate with provided params.

    params_json is a JSON string, e.g. '{"k_base":1.0,"alpha_5m":0.6,...}'
    """
    try:
        params = json.loads(params_json)
    except Exception:
        raise HTTPException(status_code=400, detail="params_json must be valid JSON")

    try:
        raw = file.file.read()
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot read CSV: {e}")

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    return simulate_df(df=df, strategy=strategy, params=params, init_servers=init_servers)


@app.post("/simulate-json")
def simulate_json(req: SimulateRequestJSON) -> Dict[str, Any]:
    """Simulate using JSON rows (useful if you don't want to upload CSV)."""
    df = pd.DataFrame(req.rows)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return simulate_df(df=df, strategy=req.strategy, params=req.params, init_servers=req.init_servers)
