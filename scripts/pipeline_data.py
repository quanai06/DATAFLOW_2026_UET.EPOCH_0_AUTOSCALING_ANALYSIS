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
        out_dir_ml="data/model_ml",
        out_dir_dl="data/model_dl"
    )

    pipe = DataPipeline(cfg)
    artifacts = pipe.run(save=True)

    df_1m_ml  = artifacts["ml"]["train"]["1m"]
    df_5m_ml  = artifacts["ml"]["train"]["5m"]
    df_15m_ml = artifacts["ml"]["train"]["15m"]

    print("train 1m:", df_1m_ml.shape);  print(df_1m_ml.head(3))
    print("train 5m:", df_5m_ml.shape);  print(df_5m_ml.head(3))
    print("train 15m:", df_15m_ml.shape); print(df_15m_ml.head(3))

    df_1m_ml  = artifacts["ml"]["test"]["1m"]
    df_5m_ml  = artifacts["ml"]["test"]["5m"]
    df_15m_ml = artifacts["ml"]["test"]["15m"]

    print("test 1m:", df_1m_ml.shape);  print(df_1m_ml.head(3))
    print("test 5m:", df_5m_ml.shape);  print(df_5m_ml.head(3))
    print("test 15m:", df_15m_ml.shape); print(df_15m_ml.head(3))



    df_1m_dl= artifacts["lstm"]["train"]["1m"]
    df_5m_dl= artifacts["lstm"]["train"]["5m"]
    df_15m_dl= artifacts["lstm"]["train"]["15m"]    

    print("train 1m:", df_1m_dl.shape);  print(df_1m_dl.head(3))
    print("train 5m:", df_5m_dl.shape);  print(df_5m_dl.head(3))
    print("train 15m:", df_15m_dl.shape); print(df_15m_dl.head(3))