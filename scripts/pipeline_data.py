import pandas as pd 
import os 
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # project root
sys.path.insert(0, str(ROOT))

from src.data_pipeline.feature_engineering import DataPipeline,PipelineConfig

if __name__ == "__main__":
    cfg = PipelineConfig(
        zip_path="../DATAFLOW_2026_UET.EPOCH_0_AUTOSCALING_ANALYSIS/data/raw/DATA-20260121T113005Z-1-001.zip",
        encoding="utf-8",
        out_dir="data/model_ml"
    )

    pipe = DataPipeline(cfg)
    artifacts = pipe.run(save=True)

    df_1m  = artifacts["train"]["1m"]
    df_5m  = artifacts["train"]["5m"]
    df_15m = artifacts["train"]["15m"]

    print("train 1m:", df_1m.shape);  print(df_1m.head(3))
    print("train 5m:", df_5m.shape);  print(df_5m.head(3))
    print("train 15m:", df_15m.shape); print(df_15m.head(3))

    df_1m  = artifacts["test"]["1m"]
    df_5m  = artifacts["test"]["5m"]
    df_15m = artifacts["test"]["15m"]

    print("test 1m:", df_1m.shape);  print(df_1m.head(3))
    print("test 5m:", df_5m.shape);  print(df_5m.head(3))
    print("test 15m:", df_15m.shape); print(df_15m.head(3))