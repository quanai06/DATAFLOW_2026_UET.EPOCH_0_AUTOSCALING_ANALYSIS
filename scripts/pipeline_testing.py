import os
import joblib
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error


TIMEFRAMES = ["1m", "5m", "15m"]
TARGETS = ["y_req_t1", "y_bytes_imp_t1"]

DATA_ML_DIR = "data/model_ml"
DATA_DL_DIR = "data/model_dl"

def load_test_data_ml(timeframe, target):
    path = f"{DATA_ML_DIR}/test_{timeframe}.parquet"
    df = pd.read_parquet(path).sort_values("timestamp")

    hold_out_date = pd.to_datetime("1995-08-16 00:00:00").tz_localize(
        df["timestamp"].dt.tz
    )
    test_df = df[df["timestamp"] >= hold_out_date].reset_index(drop=True)

    features = [
        c for c in test_df.columns
        if c not in ["timestamp", target, "y_req_t1", "y_bytes_imp_t1"]
    ]
    return test_df, features


def report_metrics(y_true, y_pred):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred)
    }


def test_xgboost():
    print("\n🚀 TESTING XGBOOST MODELS")

    model_dir = "models/xgboost"

    for tfm in TIMEFRAMES:
        for target in TARGETS:
            test_df, features = load_test_data_ml(tfm, target)

            for fname in os.listdir(model_dir):
                if not fname.startswith(f"xgboost_{target}_{tfm}"):
                    continue

                model_path = os.path.join(model_dir, fname)
                print(f"\n▶ Testing {model_path}")

                model = joblib.load(model_path)
                preds_log = model.predict(test_df[features])
                preds = np.expm1(preds_log)
                preds = np.maximum(preds, 0)

                metrics = report_metrics(test_df[target], preds)
                print(metrics)


def test_lgbm():
    print("\n🚀 TESTING LIGHTGBM MODELS")

    model_dir = "models/lgbm"

    for tfm in TIMEFRAMES:
        for target in TARGETS:
            test_df, features = load_test_data_ml(tfm, target)

            for fname in os.listdir(model_dir):
                if not fname.startswith(f"lgbm_{target}_{tfm}"):
                    continue

                model_path = os.path.join(model_dir, fname)
                print(f"\n▶ Testing {model_path}")

                model = joblib.load(model_path)
                preds = model.predict(test_df[features])

                metrics = report_metrics(test_df[target], preds)
                print(metrics)


# =========================
# LSTM TEST
# =========================
def test_lstm():
    print("\n🚀 TESTING LSTM MODELS")
    print("⚠ This may take longer and requires TensorFlow")

    for tfm in TIMEFRAMES:
        for target in TARGETS:
            data_path = f"{DATA_DL_DIR}/test_{tfm}.parquet"
            df = pd.read_parquet(data_path).sort_values("timestamp")

            hold_out_date = pd.to_datetime("1995-08-16 00:00:00").tz_localize(
                df["timestamp"].dt.tz
            )
            test_df = df[df["timestamp"] >= hold_out_date].reset_index(drop=True)

            for folder in os.listdir("models/lstm"):
                if not folder.startswith(f"lstm_{target}_{tfm}"):
                    continue

                model_dir = f"models/lstm/{folder}"
                print(f"\n▶ Testing {model_dir}")

                model = tf.keras.models.load_model(
                    f"{model_dir}/model.keras",
                    custom_objects={"pinball_loss_09": None},
                    compile=False
                )

                scaler_x = joblib.load(f"{model_dir}/scaler_x.pkl")
                scaler_y = joblib.load(f"{model_dir}/scaler_y.pkl")

                with open(f"{model_dir}/metadata.json") as f:
                    meta = json.load(f)

                features = meta["features"]
                window = meta["best_params"]["window_size"]

                X_scaled = scaler_x.transform(test_df[features])
                y_scaled = scaler_y.transform(test_df[[target]])

                X, y = [], []
                for i in range(len(X_scaled) - window):
                    X.append(X_scaled[i:i+window])
                    y.append(y_scaled[i+window])

                X, y = np.array(X), np.array(y)

                preds_scaled = model.predict(X, verbose=0)
                preds = scaler_y.inverse_transform(preds_scaled).flatten()
                y_true = scaler_y.inverse_transform(y).flatten()

                metrics = report_metrics(y_true, preds)
                print(metrics)


TESTERS = {
    "xgboost": test_xgboost,
    "lgbm": test_lgbm,
    "lstm": test_lstm
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["xgboost", "lgbm", "lstm"],
        default="lstm",
        help="Choose model type to test"
    )
    args = parser.parse_args()

    TESTERS[args.model]()
