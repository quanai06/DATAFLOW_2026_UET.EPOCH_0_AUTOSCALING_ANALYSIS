import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

class GBDTTrainer:
    def __init__(self, timeframe, target_col):
        """
        timeframe: '1m', '5m', hoặc '15m'
        target_col: 'y_req' hoặc 'y_bytes_imp'
        """
        self.timeframe = timeframe
        self.target_col = target_col
        self.model_name = f"lgbm_{target_col}_{timeframe}"
        self.results = {}

    def load_and_split(self):
        # Load data
        path = f'data/model_ml/train_{self.timeframe}.parquet'
        df = pd.read_parquet(path).sort_values('timestamp')
        
        # Chia tập Valid độc lập theo mốc 12/08
        hold_out_date = pd.to_datetime('1995-08-12 00:00:00').tz_localize(df['timestamp'].dt.tz)
        train_full = df[df['timestamp'] < hold_out_date].reset_index(drop=True)
        valid_independent = df[df['timestamp'] >= hold_out_date].reset_index(drop=True)
        
        return train_full, valid_independent

    def train(self):
        train_full, valid_independent = self.load_and_split()
        features = ['hour', 'weekday', 'is_weekend', 'flag_data_gap', 'error_rate', 'avg_url_len', 'country_nunique']
        
        print(f"\n>>> Đang huấn luyện mô hình RIÊNG BIỆT cho: {self.target_col} (Khung {self.timeframe})")
        
        tscv = TimeSeriesSplit(n_splits=5)
        fold_metrics = []
        best_model = None

        for fold, (t_idx, v_idx) in enumerate(tscv.split(train_full)):
            X_train, y_train = train_full.iloc[t_idx][features], train_full.iloc[t_idx][self.target_col]
            X_val, y_val = train_full.iloc[v_idx][features], train_full.iloc[v_idx][self.target_col]

            model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.03, random_state=42)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            
            preds = model.predict(X_val)
            fold_metrics.append([
                mean_squared_error(y_val, preds),
                np.sqrt(mean_squared_error(y_val, preds)),
                mean_absolute_error(y_val, preds),
                mean_absolute_percentage_error(y_val, preds)
            ])
            best_model = model

        # Đánh giá trên tập Valid độc lập
        final_preds = best_model.predict(valid_independent[features])
        self.results = {
            'Target': self.target_col,
            'Timeframe': self.timeframe,
            'MSE': mean_squared_error(valid_independent[self.target_col], final_preds),
            'RMSE': np.sqrt(mean_squared_error(valid_independent[self.target_col], final_preds)),
            'MAE': mean_absolute_error(valid_independent[self.target_col], final_preds),
            'MAPE': mean_absolute_percentage_error(valid_independent[self.target_col], final_preds)
        }

        # Lưu model riêng biệt vào thư mục models/
        if not os.path.exists('models'): os.makedirs('models')
        joblib.dump(best_model, f'models/{self.model_name}.pkl')
        
        return self.results