import pandas as pd
import numpy as np
import os
import joblib
import json
import itertools
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# 1. Hàm Loss được định nghĩa độc lập để tránh lỗi Serialization (Lưu model)
def pinball_loss_09(y_true, y_pred):
    quantile = 0.9
    error = y_true - y_pred
    return tf.reduce_mean(tf.maximum(quantile * error, (quantile - 1) * error))

def prepare_lstm_data(df, features, target_col, window_size, scaler_x, scaler_y, is_training=True):
    if is_training:
        x_scaled = scaler_x.fit_transform(df[features])
        y_scaled = scaler_y.fit_transform(df[[target_col]])
    else:
        x_scaled = scaler_x.transform(df[features])
        y_scaled = scaler_y.transform(df[[target_col]])
    
    xs, ys = [], []
    for i in range(len(x_scaled) - window_size):
        xs.append(x_scaled[i : (i + window_size)])
        ys.append(y_scaled[i + window_size])
    return np.array(xs), np.array(ys)

class LSTMTrainer:
    def __init__(self, timeframe, target_col, quantile=None):
        self.timeframe = timeframe
        self.target_col = target_col
        self.quantile = quantile  # Sẽ là 0.9 nếu là khung 5m
        self.model_name = f"lstm_{target_col}_{timeframe}"
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

    def load_and_split(self):
        path = f'data/model_dl/train_{self.timeframe}.parquet'
        df = pd.read_parquet(path).sort_values('timestamp')
        hold_out_date = pd.to_datetime('1995-08-16 00:00:00').tz_localize(df['timestamp'].dt.tz)
        train_full = df[df['timestamp'] < hold_out_date].reset_index(drop=True)
        valid_independent = df[df['timestamp'] >= hold_out_date].reset_index(drop=True)
        return train_full, valid_independent

    def build_model(self, input_shape, units=64, dropout=0.2, lr=0.001):
        model = Sequential([
            Input(shape=input_shape),
            LSTM(units, return_sequences=True),
            Dropout(dropout),
            LSTM(units // 2, return_sequences=False),
            Dropout(dropout),
            Dense(1)
        ])
        
        # LOGIC CHỈ DÙNG QUANTILE KHI CÓ THIẾT LẬP (Cho khung 5m)
        if self.quantile is not None:
            # Sử dụng hàm đã đặt tên thay vì lambda để Keras lưu được model
            loss_fn = pinball_loss_09 
            print(f"   [INFO] Sử dụng Pinball Loss (Quantile {self.quantile})")
        else:
            loss_fn = 'mse'
            print(f"   [INFO] Sử dụng MSE Loss")
            
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=loss_fn)
        return model

    def train(self):
        train_full, valid_independent = self.load_and_split()
        features = [c for c in train_full.columns if c not in [self.target_col, "timestamp", "y_req_t1", "y_bytes_imp_t1"]]
        
        # Grid Search nhẹ
        param_grid = {'window_size': [10, 20], 'units': [32, 64], 'lr': [0.001]}
        keys, values = zip(*param_grid.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        best_avg_rmse = float('inf')
        best_params, best_model = None, None

        print(f"\n>>> Tuning {self.model_name}...")

        for config in combinations:
            tscv = TimeSeriesSplit(n_splits=2)
            fold_rmses = []

            for fold, (t_idx, v_idx) in enumerate(tscv.split(train_full)):
                df_t, df_v = train_full.iloc[t_idx], train_full.iloc[v_idx]
                X_train, y_train = prepare_lstm_data(df_t, features, self.target_col, config['window_size'], self.scaler_x, self.scaler_y, True)
                X_val, y_val = prepare_lstm_data(df_v, features, self.target_col, config['window_size'], self.scaler_x, self.scaler_y, False)
                
                model = self.build_model((X_train.shape[1], X_train.shape[2]), units=config['units'], lr=config['lr'])
                early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
                
                model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=128, callbacks=[early_stop], verbose=0)
                
                preds = self.scaler_y.inverse_transform(model.predict(X_val, verbose=0))
                y_true = self.scaler_y.inverse_transform(y_val)
                fold_rmses.append(np.sqrt(mean_squared_error(y_true, preds)))

            avg_rmse = np.mean(fold_rmses)
            if avg_rmse < best_avg_rmse:
                best_avg_rmse, best_params, best_model = avg_rmse, config, model

        # --- LƯU TRỮ CHUẨN ---
        save_dir = f'models/lstm/{self.model_name}'
        os.makedirs(save_dir, exist_ok=True)
        
        # Lưu model (Định dạng .keras an toàn hơn)
        best_model.save(f"{save_dir}/model.keras")
        joblib.dump(self.scaler_x, f"{save_dir}/scaler_x.pkl")
        joblib.dump(self.scaler_y, f"{save_dir}/scaler_y.pkl")

        # Đánh giá cuối trên tập Independent
        self.scaler_x.fit(train_full[features])
        self.scaler_y.fit(train_full[[self.target_col]])
        combined = pd.concat([train_full.tail(best_params['window_size']), valid_independent], axis=0)
        X_test, y_test = prepare_lstm_data(combined, features, self.target_col, best_params['window_size'], self.scaler_x, self.scaler_y, False)
        
        final_preds = self.scaler_y.inverse_transform(best_model.predict(X_test, verbose=0)).flatten()
        final_y_true = self.scaler_y.inverse_transform(y_test).flatten()
        
        metadata = {
            'target': self.target_col,
            'timeframe': self.timeframe,
            'best_params': best_params,
            'metrics': {'rmse': float(np.sqrt(mean_squared_error(final_y_true, final_preds))),
                        'mae': float(mean_absolute_error(final_y_true, final_preds))},
            'features': features,
        }
        with open(f"{save_dir}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        # Lưu CSV kết quả
        pd.DataFrame({
            'timestamp': valid_independent['timestamp'].values,
            'actual': final_y_true,
            'predicted': final_preds
        }).to_csv(f'results/lstm/results_{self.model_name}.csv', index=False)

        return {
            'Target': self.target_col, 'Timeframe': self.timeframe, 
            'RMSE': metadata['metrics']['rmse'], 'Path': save_dir
        }