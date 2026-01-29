import pandas as pd
import numpy as np
import os
import joblib

class DataProcessor:
    def __init__(self, input_dir="data/processed", output_dir="data/model_ml"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        # Đảm bảo thư mục đầu ra tồn tại
        os.makedirs(self.output_dir, exist_ok=True)

    def load_aggregated_data(self, timeframe='5m'):
        """Đọc dữ liệu từ kết quả của file aggregation notebook"""
        file_path = os.path.join(self.input_dir, f"train_{timeframe}.parquet")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Không tìm thấy file: {file_path}. Hãy chạy notebook aggregation trước.")
        return pd.read_parquet(file_path)

    def apply_feature_engineering(self, df, target_col='y_req'):
        """
        Thực hiện chuẩn hóa cuối cùng và tạo lag features 
        Dựa trên các cột bạn đã có: hour, weekday, flag_data_gap...
        """
        # Sắp xếp theo thời gian để tạo lag không bị sai [cite: 21]
        df = df.sort_values('timestamp')

        # Tạo Lag Features (Biến trễ) - Quan trọng cho GBDT
        for lag in [1, 2, 3]:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        # Tạo Rolling features (Trung bình trượt)
        df[f'{target_col}_rolling_mean_3'] = df[target_col].shift(1).rolling(window=3).mean()

        # Xử lý flag_data_gap: Đảm bảo kiểu dữ liệu chuẩn
        if 'flag_data_gap' in df.columns:
            df['flag_data_gap'] = df['flag_data_gap'].astype('int8')

        # Loại bỏ các dòng NaN do tạo lag
        return df.dropna()

    def save_for_training(self, df, timeframe='5m'):
        """Lưu dữ liệu cuối cùng vào thư mục model_ml"""
        output_path = os.path.join(self.output_dir, f"train_{timeframe}_final.parquet")
        df.to_parquet(output_path)
        print(f"--- Đã chuẩn bị xong dữ liệu {timeframe} tại: {output_path} ---")

    def process_all(self):
        """Chạy cho cả 3 khung thời gian yêu cầu [cite: 37, 38, 39]"""
        for tf in ['1m', '5m', '15m']:
            df = self.load_aggregated_data(tf)
            df_final = self.apply_feature_engineering(df)
            self.save_for_training(df_final, tf)

if __name__ == "__main__":
    processor = DataProcessor()
    processor.process_all()