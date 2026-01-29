import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import xgboost as xgb
import os

class XGBoostTrainer:
    def __init__(self, timeframe, target_col):
        self.timeframe = timeframe
        self.target_col = target_col
        self.model_name = f"xgb_{target_col}_{timeframe}"
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

    def calculate_mape(self,y_true, y_pred):
        # Chỉ tính tại những thời điểm thực tế > 1
        mask = y_true > 1 
        if np.any(mask):
            # Tính MAPE trên tập dữ liệu đã lọc
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
            return mape
        return 0

    def train(self):
        train_full, valid_independent = self.load_and_split()
        
        # Thêm các features 
        features = [col for col in train_full.columns if col not in ['timestamp', self.target_col]]

        # Đảm bảo target_col (ví dụ y_req_t1) KHÔNG nằm trong features
        features = [f for f in features if f != self.target_col]

        X_valid_ind = valid_independent[features]
        y_valid_ind = valid_independent[self.target_col]

        tscv = TimeSeriesSplit(n_splits=5)
        best_model = None

        for fold, (t_idx, v_idx) in enumerate(tscv.split(train_full)):
            X_train, y_train = train_full.iloc[t_idx][features], train_full.iloc[t_idx][self.target_col]
            X_val, y_val = train_full.iloc[v_idx][features], train_full.iloc[v_idx][self.target_col]

            # LOG-TRANSFORM CHO Y TRƯỚC KHI TRAIN
            # Dùng log1p để tránh lỗi ln(0) vì log1p(x) = ln(1+x)
            y_train_log = np.log1p(y_train)
            y_val_log = np.log1p(y_val)

            # CHUYỂN SANG XGBOOST TẠI ĐÂY
            model = xgb.XGBRegressor(
                n_estimators=1000,
                learning_rate=0.02,
                max_depth=8,
                early_stopping_rounds=50,
                tree_method='hist', # Tăng tốc độ train
                random_state=42
            )
            
            model.fit(
                X_train, y_train_log, 
                eval_set=[(X_val, y_val_log)],
                verbose=False
            )
            best_model = model

        # BƯỚC 2: DỰ BÁO VÀ ĐƯA VỀ GIÁ TRỊ GỐC (EXP)
        preds_log = best_model.predict(X_valid_ind)
        # Dùng expm1 để nghịch đảo của log1p: expm1(x) = e^x - 1
        preds = np.expm1(preds_log)
        
        # Đảm bảo không có giá trị dự báo âm (do sai số exp)
        preds = np.maximum(preds, 0)

        # Tính toán các chỉ số đề bài yêu cầu
        self.results = {
            'Model_Type': 'XGBoost',
            'Target': self.target_col,
            'Timeframe': self.timeframe,
            'RMSE': np.sqrt(mean_squared_error(y_valid_ind, preds)),
            'MSE': mean_squared_error(y_valid_ind, preds),
            'MAE': mean_absolute_error(y_valid_ind, preds),
            'MAPE': self.calculate_mape(y_valid_ind, preds)
        }
        # LƯU KẾT QUẢ VÀ MODEL
        # 1. Tạo thư mục nếu chưa có
        if not os.path.exists('models/xgboost'): os.makedirs('models/xgboost')
        if not os.path.exists('results/xgboost'): os.makedirs('results/xgboost')

        # 2. Lưu kết quả dự báo (.csv) vào thư mục results
        # File này phục vụ cho việc tính toán Optimizer và viết báo cáo
        csv_path = f'results/xgboost/results_{self.model_name}.csv'
        output = valid_independent[['timestamp', self.target_col]].copy()
        output['predicted'] = preds
        output.to_csv(csv_path, index=False)
        print(f"Đã lưu kết quả dự báo tại: {csv_path}")

        # 3. Lưu bộ não (.pkl) vào thư mục models
        # File này phục vụ cho việc làm Demo/API sau này
        model_path = f'models/xgboost/{self.model_name}.pkl'
        joblib.dump(best_model, model_path)
        print(f"Đã lưu model tại: {model_path}")

        return self.results