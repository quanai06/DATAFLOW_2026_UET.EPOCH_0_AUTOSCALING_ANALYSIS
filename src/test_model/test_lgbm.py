import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error

def run_test_inference(model_path, timeframe, target_col):
    """
    Thực hiện dự báo trên tập test cho model LGBM.
    model_path: Đường dẫn đến file .pkl của model
    timeframe: '1m', '5m', hoặc '15m'
    target_col: 'y_req_t1' hoặc 'y_bytes_imp_t1'
    """
    print(f"\n--- Đang thực hiện Inference LGBM cho: {target_col} | {timeframe} ---")
    
    # 1. Kiểm tra sự tồn tại của model
    if not os.path.exists(model_path):
        print(f"❌ Không tìm thấy model tại: {model_path}")
        return

    # 2. Load model
    model = joblib.load(model_path)
    
    # 3. Load dữ liệu test thực tế từ disk
    # Lưu ý: folder data/model_ml/ theo cấu trúc
    test_path = f'data/model_ml/test_{timeframe}.parquet'
    
    if not os.path.exists(test_path):
        print(f"❌ Không tìm thấy file test tại: {test_path}")
        return
        
    df_test = pd.read_parquet(test_path).sort_values('timestamp')
    
    # 4. Xác định Features
    # Loại bỏ các cột target và timestamp để lấy feature đầu vào giống lúc train
    features = [c for c in df_test.columns if c not in [target_col, "timestamp", "y_req_t1", "y_bytes_imp_t1"]]
    
    X_test = df_test[features]
    y_true = df_test[target_col].values
    
    # 5. Dự báo
    preds = model.predict(X_test)
    
    # 6. Tính toán metrics trên tập Test
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    mae = mean_absolute_error(y_true, preds)
    
    # Tính MAPE (tránh chia cho 0)
    mape = np.mean(np.abs((y_true - preds) / np.where(y_true == 0, 1, y_true))) * 100
    
    print(f"SỐ LIỆU TẬP TEST ({timeframe}):")
    print(f" - RMSE: {rmse:.4f}")
    print(f" - MAE:  {mae:.4f}")
    print(f" - MAPE: {mape:.2f}%")
    
    # 7. Lưu kết quả ra file CSV
    results_df = pd.DataFrame({
        'timestamp': df_test['timestamp'].values,
        'actual': y_true,
        'predicted': preds
    })
    
    # Tạo folder results nếu chưa có
    output_dir = "results/lgbm"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = f"{output_dir}/results_lgbm_test_{target_col}_{timeframe}.csv"
    results_df.to_csv(output_path, index=False)
    print(f"✅ Đã lưu kết quả tại: {output_path}")

if __name__ == "__main__":
    # Cấu hình danh sách cần chạy
    timeframes = ['1m', '5m', '15m']
    targets = ['y_req_t1', 'y_bytes_imp_t1']
    
    for tf in timeframes:
        for target in targets:
            # Đường dẫn file model .pkl (theo logic lưu file của class LGBMTrainer)
            MODEL_FILE = f"models/lgbm/lgbm_{target}_{tf}.pkl"
            
            try:
                run_test_inference(
                    model_path=MODEL_FILE, 
                    timeframe=tf, 
                    target_col=target
                )
            except Exception as e:
                print(f"❌ Lỗi khi xử lý {target} {tf}: {e}")