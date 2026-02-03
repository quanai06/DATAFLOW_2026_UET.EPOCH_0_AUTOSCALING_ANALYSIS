import pandas as pd
import os
import sys
import argparse
import numpy as np
import random
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train_model.train_lgbm import LGBMTrainer
from src.train_model.train_xgboost import XGBoostTrainer
from src.train_model.train_lstm import LSTMTrainer

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

timeframes = ['1m', '5m', '15m']
targets = ['y_req_t1', 'y_bytes_imp_t1']
all_results = []

def train_lgbm():
    for tf in timeframes:
        for target in targets:
            try:
                q_val = None
                if tf == '5m' or tf == '1m':
                    q_val = 0.9

                trainer = LGBMTrainer(timeframe=tf, target_col=target, quantile=q_val)
                metrics = trainer.train()
                print(f"Hoàn thành LGBM: {target} | {tf}")
                all_results.append(metrics)
            except Exception as e:
                print(f"Lỗi LGBM {target} {tf}: {e}")
    
    if all_results:
        summary_df = pd.DataFrame(all_results)
        print("\n" + "="*50)
        print("BẢNG TỔNG HỢP KẾT QUẢ (BENCHMARKING)")
        print("="*50)
        print(summary_df)
        summary_df.to_csv('results/lgbm/lgbm_performance_report.csv', index=False)
        

def train_xgboost():
    for tf in timeframes:
        for tg in targets:
            try:
                trainer = XGBoostTrainer(tf, tg)
                metrics = trainer.train()
                all_results.append(metrics)
                if tf in ['5m', '1m']:
                    trainer_q90 = XGBoostTrainer(tf, tg, objective='reg:quantileerror', quantile_alpha=0.9)
                    metrics_q90 = trainer_q90.train()
                    all_results.append(metrics_q90)
                    
            except Exception as e:
                print(f"Lỗi khi train {tg} khung {tf}: {e}")
                
    if all_results:
        summary_df = pd.DataFrame(all_results)
        print("\n" + "="*50)
        print("BẢNG TỔNG HỢP KẾT QUẢ (BENCHMARKING)")
        print("="*50)
        print(summary_df)
        summary_df.to_csv('results/xgboost/xgboost_performance_report.csv', index=False)

def train_lstm():
    for tf in timeframes:
        for target in targets:
            try:
                q_val = None
                if tf == '5m' or tf == '1m':
                    q_val = 0.9
                    
                trainer = LSTMTrainer(timeframe=tf, target_col=target, quantile=q_val)
                metrics = trainer.train()
                all_results.append(metrics)
                print(f"Hoàn thành LSTM: {target} | {tf}")
            except Exception as e:
                print(f"Lỗi LSTM {target} {tf}: {e}")
    
    if all_results:
        summary_df = pd.DataFrame(all_results)
        print("\n" + "="*50)
        print("BẢNG TỔNG HỢP KẾT QUẢ (BENCHMARKING)")
        print("="*50)
        print(summary_df)
        summary_df.to_csv('results/lstm/lstm_performance_report.csv', index=False)


TRAINERS = {
    "xgboost": train_xgboost,
    "lgbm": train_lgbm,
    "lstm": train_lstm
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["xgboost", "lgbm", "lstm", "all"],
        default="lstm"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "load"],
        default="load"
    )
    args = parser.parse_args()

    if args.mode == "train":
        if args.model == "all":
            for fn in TRAINERS.values():
                fn()
        else:
            TRAINERS[args.model]()
    else:
        print("⚡ LOAD MODE: Skip training, use pretrained models")