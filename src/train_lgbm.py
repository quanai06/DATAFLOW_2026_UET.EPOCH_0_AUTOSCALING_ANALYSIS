import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import RandomizedSearchCV

class LGBMTrainer:
    def __init__(self, timeframe, target_col, quantile=None):
        """
        timeframe: '1m', '5m', hoặc '15m'
        target_col: 'y_req_t1' hoặc 'y_bytes_imp_t1'
        """
        self.timeframe = timeframe
        self.target_col = target_col
        self.quantile = quantile
        self.model_name = f"lgbm_{target_col}_{timeframe}"
        self.results = {}

    def load_and_split(self):
        # Load data
        path = f'data/model_ml/train_{self.timeframe}.parquet'
        df = pd.read_parquet(path).sort_values('timestamp')
        
        # Chia tập Valid độc lập theo mốc 16/08
        hold_out_date = pd.to_datetime('1995-08-16 00:00:00').tz_localize(df['timestamp'].dt.tz)
        train_full = df[df['timestamp'] < hold_out_date].reset_index(drop=True)
        valid_independent = df[df['timestamp'] >= hold_out_date].reset_index(drop=True)
        
        return train_full, valid_independent

    def train(self):
        train_full, valid_independent = self.load_and_split()
        features = [c for c in train_full.columns if c not in [self.target_col, "timestamp", "y_req_t1", "y_bytes_imp_t1"]]
        
        X_train_full = train_full[features]
        y_train_full = train_full[self.target_col]

        print(f"\n>>> Đang tìm tham số tối ưu (RandomSearch) cho: {self.target_col} ({self.timeframe})")

        # 1. Thiết lập lưới tham số (Bạn có thể thêm bớt tùy sức mạnh máy tính)
        param_grid = {
            'n_estimators': [500, 1000],
            'learning_rate': [0.01, 0.05],
            'num_leaves': [31, 63],
            'feature_fraction': [0.8, 0.9],
            'objective': ['quantile' if self.quantile else 'regression'],
            'alpha': [self.quantile if self.quantile else 0.5],
            'random_state': [42],
            'verbosity': [-1] # Tắt log thừa của LGBM
        }

        # 2. Khởi tạo TimeSeriesSplit để đảm bảo tính thời gian
        tscv = TimeSeriesSplit(n_splits=5)

        # 3. Khởi tạo model và GridSearchCV
        lgbm = lgb.LGBMRegressor()
        
        random_search = RandomizedSearchCV(
            estimator=lgbm,
            param_distributions=param_grid,
            n_iter=10,
            cv=tscv,
            n_jobs=-1,
            verbose=1
        )

        # 4. Huấn luyện
        random_search.fit(X_train_full, y_train_full)
        
        best_model = random_search.best_estimator_
        print(f"   🔥 Tham số tốt nhất: {random_search.best_params_}")

        # --- Đánh giá trên tập Valid độc lập (16/08 trở đi) ---
        final_preds = best_model.predict(valid_independent[features])
        final_y_true = valid_independent[self.target_col].values

        # Lưu kết quả CSV
        results_df = pd.DataFrame({
            'timestamp': valid_independent['timestamp'].values,
            self.target_col: final_y_true,
            'predicted': final_preds
        })
        
        os.makedirs('results/lgbm', exist_ok=True)
        csv_path = f'results/lgbm/results_lgbm_{self.target_col}_{self.timeframe}.csv'
        results_df.to_csv(csv_path, index=False)
        print(f"   ✅ Đã lưu CSV: {csv_path}")

        # Lưu Model tốt nhất
        os.makedirs('models/lgbm', exist_ok=True)
        joblib.dump(best_model, f'models/lgbm/{self.model_name}.pkl')
        
        # Kết quả cuối cùng
        self.results = {
            'Target': self.target_col,
            'Timeframe': self.timeframe,
            'Best_Params': random_search.best_params_,
            'RMSE': np.sqrt(mean_squared_error(final_y_true, final_preds)),
            'MAE': mean_absolute_error(final_y_true, final_preds)
        }
        
        return self.results