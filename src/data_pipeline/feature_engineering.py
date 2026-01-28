# data_pipeline.py
from __future__ import annotations

import os
import re
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# =========================
# Utils
# =========================
def _entropy(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) == 0:
        return 0.0
    p = s.value_counts(normalize=True)
    return float(-(p * np.log2(p + 1e-12)).sum())

def _top_share(s: pd.Series, exclude=("other_country",)) -> float:
    s = s.dropna()
    s1 = s[~s.isin(exclude)]
    if len(s) == 0 or len(s1) == 0:
        return 0.0
    return float(s1.value_counts().iloc[0] / len(s))


# =========================
# Stage 1: Ingest + Parse raw log
# =========================
class LogZipIngestor:
    """Read raw logs from a .zip containing DATA/train.txt and DATA/test.txt (like your notebook)."""

    def __init__(self, zip_path: str, encoding: str = "utf-8"):
        self.zip_path = zip_path
        self.encoding = encoding

    def read_text(self, inner_path: str) -> str:
        with zipfile.ZipFile(self.zip_path, "r") as z:
            with z.open(inner_path, "r") as f:
                return f.read().decode(self.encoding, errors="replace")

    def read_train_test_lines(self,
                              train_inner_path: str = "DATA/train.txt",
                              test_inner_path: str = "DATA/test.txt") -> Tuple[List[str], List[str]]:
        train_text = self.read_text(train_inner_path)
        test_text = self.read_text(test_inner_path)
        return train_text.splitlines(), test_text.splitlines()


class LogParser:
    """Regex parse: host, timestamp, request, response, bytes (same as notebook)."""

    def __init__(self):
        self.pattern = re.compile(
            r'(?P<host>\S+) .* '
            r'\[(?P<timestamp>[^\]]+)\] '
            r'"(?P<request>[^"]+)" '
            r'(?P<response>\d{3}) '
            r'(?P<bytes>\d+|-)'
        )

    def to_dataframe(self, lines: List[str]) -> pd.DataFrame:
        rows = []
        for line in lines:
            m = self.pattern.search(line)
            if not m:
                continue
            row = m.groupdict()
            row["bytes"] = None if row["bytes"] == "-" else int(row["bytes"])
            rows.append(row)
        return pd.DataFrame(rows)


# =========================
# Stage 2: Clean + enrich row-level features
# =========================
class RowCleaner:
    """Apply dtype, parse timestamp, add host/request/response/bytes features """

    COLS_RAW = ["host", "timestamp", "request", "response", "bytes"]

    HTTP_METHODS = ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'HEAD', 'OPTIONS', 'TRACE', 'CONNECT']
    IMAGE_TYPES = ['gif', 'jpg', 'jpeg', 'tiff', 'png', 'bmp', 'xbm', 'eps', 'art', 'ps']
    AV_TYPES = ['wav', 'avi', 'mp3', 'wma', 'mpg', 'ksc']
    FILE_TYPES = ['doc', 'rtf', 'xls', 'txt', 'bak', 'xb', 'pdf', 'sta', 'bps', 'new', 'zip', 'bad', 'out', 'orig','dat', 'lists']
    PAGE_TYPES = ['html', 'pl', 'htm', 'map', 'perl', 'dec', 'software']

    SWITCHER_DOMAIN = {
        'net': 'network', 'com': 'commercial', 'org': 'organisation',
        'edu': 'education', 'gov': 'government', 'int': 'international', 'mil': 'military'
    }

    def apply_data_type(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.COLS_RAW if c not in df.columns]
        if missing:
            raise ValueError(f"Thiếu cột: {missing}")

        d = df[self.COLS_RAW].copy()

        # timestamp: strip timezone -0400 using extract (as notebook)
        ts = d["timestamp"].astype(str).str.strip()
        ts = ts.str.extract(r'(\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2})', expand=False)
        d["timestamp"] = pd.to_datetime(ts, format="%d/%b/%Y:%H:%M:%S", errors="coerce")

        # response numeric-friendly
        d["response"] = pd.to_numeric(d["response"], errors="coerce")

        # bytes numeric
        d["bytes"] = pd.to_numeric(d["bytes"], errors="coerce")
        d["bytes"] =d["bytes"]/1000 #sang KB

        # cast text columns
        d["host"] = d["host"].astype("string")
        d["request"] = d["request"].astype("string")

        return d

    # ---- host features (tối giản để phục vụ aggregation notebook) ----
    def _get_tld(self, host: str) -> str | None:
        if not isinstance(host, str) or host == "":
            return None
        # ip => None
        if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", host):
            return None
        parts = host.rsplit(".", 1)
        if len(parts) != 2:
            return None
        tld = parts[-1].lower()
        return None if tld.isdigit() else tld

    def _categorise_visitor(self, tld: str | None) -> str:
        if tld is None or (isinstance(tld, float) and np.isnan(tld)):
            return "unknown"
        tld = str(tld)

        if len(tld) == 2:
            return "country"
        if len(tld) == 3:
            return self.SWITCHER_DOMAIN.get(tld, "unknown")
        return "unknown"

    def add_host_features(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["tld"] = d["host"].astype(str).apply(self._get_tld)
        d["visitor_category"] = d["tld"].apply(self._categorise_visitor)

        # aggregation expects these:
        d["visitor_country"] = np.where(d["visitor_category"] == "country", d["tld"], "other_country")
        d["is_commercial"] = (d["visitor_category"] == "commercial").astype("int8")
        d["is_unknown"] = (d["visitor_category"] == "unknown").astype("int8")
        return d

    # ---- request features (the same as 05_request.ipynb, no plots) ----
    def _parse_request_basic(self, request: str) -> Tuple[str, str]:
        parts = str(request).split(" ")
        method = parts[0] if len(parts) > 0 else "UNKNOWN"
        url = parts[1] if len(parts) > 1 else "UNKNOWN"
        return method, url

    def _get_resource_details(self, url: str) -> Tuple[str, str]:
        try:
            url_clean = str(url).replace("HTTP/1.0", "").strip()
            last_slash = url_clean.rindex("/")
            directory = url_clean[: last_slash + 1]
            resource = url_clean[last_slash + 1 :]
        except Exception:
            directory = url
            resource = ""
        return directory, resource

    def _get_resource_type(self, resource: str) -> str:
        try:
            dot = str(resource).rindex(".")
            ext = str(resource)[dot + 1 :].lower()
            if ext in self.IMAGE_TYPES:
                return "Image"
            if ext in self.AV_TYPES:
                return "Audio/Video"
            if ext in self.FILE_TYPES:
                return "File"
            if ext in self.PAGE_TYPES:
                return "Page"
            return "UNKNOWN"
        except Exception:
            return "Directory"

    def _get_path_depth(self, directory: str) -> int:
        if directory == "UNKNOWN":
            return -1
        return str(directory).count("/")

    def _get_url_length(self, url: str) -> int:
        return len(str(url))

    def add_request_features(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()

        basic = d["request"].astype(str).apply(self._parse_request_basic)
        d["reqmethod"] = [x[0] for x in basic]
        d["requrl"] = [x[1] for x in basic]

        d["reqmethod"] = d["reqmethod"].apply(lambda x: x if x in self.HTTP_METHODS else "UNKNOWN")

        details = d["requrl"].apply(self._get_resource_details)
        d["reqdirectory"] = [x[0] for x in details]
        d["reqresource"] = [x[1] for x in details]

        d["reqresourcetype"] = d["reqresource"].apply(self._get_resource_type)
        d["reqpathdepth"] = d["reqdirectory"].apply(self._get_path_depth)

        dynamic_exts = ["pl", "cgi", "perl"]
        d["is_dynamic"] = d["reqresource"].apply(
            lambda x: 1 if any(str(x).lower().endswith(ext) for ext in dynamic_exts) else 0
        ).astype("int8")

        d["requrllength"] = d["requrl"].apply(self._get_url_length).astype("int32")

        # one-hot used in aggregation notebook
        d["is_heavy"] = d["reqresourcetype"].isin(["Audio/Video", "File"]).astype("int8")
        d["is_image"] = (d["reqresourcetype"] == "Image").astype("int8")
        d["is_get"] = (d["reqmethod"] == "GET").astype("int8")
        d["is_post"] = (d["reqmethod"] == "POST").astype("int8")

        return d

    # ---- response features ----
    def add_response_features(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()

        def response_class(code) -> str:
            try:
                k = int(str(int(code))[0])
            except Exception:
                return "unknown"
            return {
                1: "informational",
                2: "successful",
                3: "redirection",
                4: "client error",
                5: "server error",
            }.get(k, "unknown")

        d["response_class"] = d["response"].apply(response_class)
        return d

    # ---- bytes imputation features (from aggregation notebook) ----
    def add_bytes_imputed(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["bytes_missing"] = d["bytes"].isna().astype("int8")
        d["bytes_imputed"] = d["bytes"]

        # median bytes by status
        med = d.groupby("response")["bytes"].median()

        # 3xx => 0
        m3 = (d["response_class"] == "redirection") & d["bytes_imputed"].isna()
        d.loc[m3, "bytes_imputed"] = 0

        # 2xx/4xx/5xx => fill median by response
        for lo, hi in [(200, 299), (400, 499), (500, 599)]:
            m = d["response"].between(lo, hi) & d["bytes_imputed"].isna()
            d.loc[m, "bytes_imputed"] = d.loc[m, "response"].map(med).fillna(0)

        d["bytes_imputed"] = pd.to_numeric(d["bytes_imputed"], errors="coerce").fillna(0)
        return d

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        d = self.apply_data_type(df)
        d = self.add_host_features(d)
        d = self.add_request_features(d)
        d = self.add_response_features(d)
        d = self.add_bytes_imputed(d)
        return d


# =========================
# Stage 3: Aggregate to 1m/5m/15m tables
# =========================
class Aggregator:
    def make_table(self, df_row: pd.DataFrame, freq: str) -> pd.DataFrame:
        d = df_row.copy()
        d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
        d = d.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")

        # numeric safety
        d["response"] = pd.to_numeric(d["response"], errors="coerce")
        d["bytes"] = pd.to_numeric(d["bytes"], errors="coerce")
        d["bytes_imputed"] = pd.to_numeric(d["bytes_imputed"], errors="coerce")

        r = d.resample(freq)
        out = pd.DataFrame(index=r.size().index)

        # targets at time t (aggregation notebook used y_req, y_bytes_imp as current)
        out["y_req"] = r.size().astype("int64")
        out["y_bytes_imp"] = r["bytes_imputed"].sum()

        # bytes quality
        out["bytes_missing_rate"] = r["bytes_missing"].mean().fillna(0.0)
        out["bytes_all_missing"] = (out["bytes_missing_rate"] == 1.0).astype("int8")

        # health / response mix
        out["error_rate"] = (d["response"] >= 400).resample(freq).mean().fillna(0.0)
        out["server_error_rate"] = (d["response"] >= 500).resample(freq).mean().fillna(0.0)
        out["redirection_rate"] = ((d["response"] >= 300) & (d["response"] < 400)).resample(freq).mean().fillna(0.0)

        # mix
        out["dynamic_rate"] = r["is_dynamic"].mean().fillna(0.0)
        out["commercial_rate"] = r["is_commercial"].mean().fillna(0.0)
        out["unknown_rate"] = r["is_unknown"].mean().fillna(0.0)

        # url/path
        out["avg_path_depth"] = r["reqpathdepth"].mean()

        # diversity / concentration
        out["country_nunique"] = r["visitor_country"].nunique()
        out["dir_nunique"] = r["reqdirectory"].nunique()
        out["method_nunique"] = r["reqmethod"].nunique()
        out["top_country_share"] = r["visitor_country"].apply(_top_share).fillna(0.0)
        out["endpoint_entropy"] = r["reqdirectory"].apply(_entropy).fillna(0.0)

        # content/behavior ratios
        out["ratio_heavy"] = r["is_heavy"].mean().fillna(0.0)
        out["ratio_image"] = r["is_image"].mean().fillna(0.0)
        out["ratio_get"] = r["is_get"].mean().fillna(0.0)
        out["ratio_post"] = r["is_post"].mean().fillna(0.0)

        # time features
        out = out.reset_index()
        out["hour"] = out["timestamp"].dt.hour
        out["weekday"] = out["timestamp"].dt.weekday
        out["is_weekend"] = (out["weekday"] >= 5).astype("int8")

        return out

    def make_multi_freq(self, df_row: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        return {
            "1m": self.make_table(df_row, "1min"),
            "5m": self.make_table(df_row, "5min"),
            "15m": self.make_table(df_row, "15min"),
        }



# =========================
# Stage 4: ML-ready features (gap flag, time cyclical, shift labels t+1, lags/rolling)
# =========================
class MLFeatureBuilder:
    def flag_data_gap(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["flag_data_gap"] = d["avg_path_depth"].isna().astype("int8")
        return d

    def fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["avg_path_depth"] = d["avg_path_depth"].fillna(0)
        return d

    def add_cyclical_time(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["sin_hour"] = np.sin(2 * np.pi * d["hour"] / 24.0)
        d["cos_hour"] = np.cos(2 * np.pi * d["hour"] / 24.0)
        d["sin_weekday"] = np.sin(2 * np.pi * d["weekday"] / 7.0)
        d["cos_weekday"] = np.cos(2 * np.pi * d["weekday"] / 7.0)
        return d

    def add_targets_t1(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.sort_values("timestamp").copy()
        d["y_req_t1"] = d["y_req"].shift(-1)
        d["y_bytes_imp_t1"] = d["y_bytes_imp"].shift(-1)
        d = d.dropna(subset=["y_req_t1","y_bytes_imp_t1"])

        return d
    def select_features(self,df:pd.DataFrame,target_cols :List[str] =["y_req_t1","y_bytes_imp_t1"],gap_flag_col :str="flag_data_gap",)-> pd.DataFrame:
        candidate_features=[
            "timestamp",
            *target_cols,

            "y_req",
            "y_bytes_imp",

            # Quality / missingness
            "bytes_missing_rate",
            "bytes_all_missing",
                    
            # Rates
            "error_rate", 
            "server_error_rate", 
            "redirection_rate",
            "dynamic_rate", 
            "commercial_rate", 
            "unknown_rate",
                    
            # URL/Path
            "avg_path_depth",
                    
            # Diversity/Geo/Method
            "country_nunique", 
            "dir_nunique", 
            "method_nunique",
            "top_country_share", 
            "endpoint_entropy",
                    
            # Time features
            "hour", 
            "weekday", 
            "is_weekend",
            "sin_hour", 
            "cos_hour",
            "sin_weekday",
            "cos_weekday",
                    
            # Gap flag
            gap_flag_col,
            ]
        print(f"select {len(candidate_features)} columns")
        df=df[candidate_features]
        return df

    def add_lag_rolling_vel_acc(self, df: pd.DataFrame, freq_key: str) -> pd.DataFrame:
        """
        Based on your notebook:
          - add avg_payload
          - lags/rolling(mean,std) on shifted(1)
          - vel/acc
          - mask gaps => NaN before generating features
        """
        d = df.sort_values("timestamp").copy()

        # derived
        d["avg_payload"] = d["y_bytes_imp"] / (d["y_req"] + 1e-6)

        cols = [
            "y_req", "y_bytes_imp", "bytes_missing_rate",
            "error_rate", "top_country_share", "endpoint_entropy"
        ]

        # mask gaps
        if "flag_data_gap" in d.columns:
            mask_gap = d["flag_data_gap"] == 1
            d.loc[mask_gap, cols] = np.nan

        cfg = {
            "1m":  {"lags": (1, 2, 3, 5, 15), "windows": (5, 15, 30)},
            "5m":  {"lags": (1, 2, 3, 6),      "windows": (6, 12, 24)},
            "15m": {"lags": (1, 2, 3),         "windows": (3, 6, 12)},
        }
        lags = cfg[freq_key]["lags"]
        windows = cfg[freq_key]["windows"]

        new_cols = {}

        # lags
        for c in cols:
            if c not in d.columns:
                continue
            for l in lags:
                new_cols[f"{c}_lag{l}"] = d[c].shift(l)

        # rolling mean/std on shifted(1)
        for c in cols:
            if c not in d.columns:
                continue
            s = d[c].shift(1)
            for w in windows:
                new_cols[f"{c}_rmean{w}"] = s.rolling(w, min_periods=1).mean()
                new_cols[f"{c}_rstd{w}"] = s.rolling(w, min_periods=2).std()

        # vel/acc
        for c in cols:
            if c not in d.columns:
                continue
            s_prev = d[c].shift(1)
            vel = s_prev.diff(1)
            new_cols[f"{c}_vel"] = vel
            new_cols[f"{c}_acc"] = vel.diff(1)

        d = pd.concat([d, pd.DataFrame(new_cols, index=d.index)], axis=1)

        d = d[d["flag_data_gap"] == 0].copy()
        return d

    def build(self, df_ts: pd.DataFrame, freq_key: str) -> pd.DataFrame:
        d = df_ts.copy()
        d = self.flag_data_gap(d)
        d = self.fill_missing(d)
        d = self.add_cyclical_time(d)
        d = self.add_targets_t1(d)
        d = self.select_features(d)
        d = self.add_lag_rolling_vel_acc(d, freq_key=freq_key)
        return d


# =========================
# Orchestrator: full pipeline (before model)
# =========================
@dataclass
class PipelineConfig:
    zip_path: str
    encoding: str = "utf-8"
    out_dir: str = "data/gold"   # where parquet outputs go


class DataPipeline:
    """End-to-end: zip -> parsed rows -> cleaned rows -> aggregated 1m/5m/15m -> ML-ready features."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.ingestor = LogZipIngestor(cfg.zip_path, encoding=cfg.encoding)
        self.parser = LogParser()
        self.cleaner = RowCleaner()
        self.aggregator = Aggregator()
        self.ml_builder = MLFeatureBuilder()

    def run(self, save: bool = True) -> Dict[str, Dict[str, pd.DataFrame]]:
        train_lines, test_lines = self.ingestor.read_train_test_lines()

        df_train_raw = self.parser.to_dataframe(train_lines)
        df_test_raw = self.parser.to_dataframe(test_lines)
        print ("ĐÃ CHUYỂN THÀNH DATAFRAME-------")

        df_train_row = self.cleaner.run(df_train_raw)
        df_test_row = self.cleaner.run(df_test_raw)
        print("XỬ LÍ XONG CỘT-------")

        train_multi = self.aggregator.make_multi_freq(df_train_row)
        test_multi = self.aggregator.make_multi_freq(df_test_row)
        print ("HOÀN THÀNH 99%")

        train_ml = {k: self.ml_builder.build(v, k) for k, v in train_multi.items()}
        test_ml = {k: self.ml_builder.build(v, k) for k, v in test_multi.items()}

        if save:
            os.makedirs(self.cfg.out_dir, exist_ok=True)
            for k, dfk in train_ml.items():
                dfk.to_parquet(os.path.join(self.cfg.out_dir, f"train_{k}.parquet"), index=False)
            for k, dfk in test_ml.items():
                dfk.to_parquet(os.path.join(self.cfg.out_dir, f"test_{k}.parquet"), index=False)

        return {"train": train_ml, "test": test_ml}