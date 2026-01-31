import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error

def run_test_inference(model_path, timeframe, target_col):
    """
    Thực hiện dự báo trên tập test cho model XGBoost.
    Tự động chạy cả model Mean và model Q90 (nếu có) và lưu ra 2 file riêng.
    """
    print(f"\n--- Đang thực hiện Inference XGBoost cho: {target_col} | {timeframe} ---")
    
    # 1. Load dữ liệu test thực tế từ disk (Dùng chung cho cả Mean và Q90)
    test_path = f'data/model_ml/test_{timeframe}.parquet'
    
    if not os.path.exists(test_path):
        print(f"❌ Không tìm thấy file test tại: {test_path}")
        return
        
    df_test = pd.read_parquet(test_path).sort_values('timestamp')
    
    # 2. Xác định Features (Logic giống hệt LGBM mẫu)
    features = [col for col in df_test.columns if col not in ['timestamp', target_col]]
    
    X_test = df_test[features]
    y_true = df_test[target_col].values
    
    # Tạo folder output chung
    output_dir = "results/xgboost"
    os.makedirs(output_dir, exist_ok=True)

    # ==============================================================================
    # PHẦN 1: XỬ LÝ MODEL MEAN
    # ==============================================================================
    if os.path.exists(model_path):
        print(f"🔹 Đang chạy model MEAN...")
        model_mean = joblib.load(model_path)
        
        # Dự báo (Có xử lý ngược logarit nếu model train bằng log)
        preds_log = model_mean.predict(X_test)
        preds_mean = np.expm1(preds_log) # Đảo ngược log
        preds_mean = np.maximum(preds_mean, 0) # Đảm bảo không âm
        
        # Tính toán metrics
        rmse = np.sqrt(mean_squared_error(y_true, preds_mean))
        mae = mean_absolute_error(y_true, preds_mean)
        # Tính MAPE giống logic LGBM mẫu
        mape = np.mean(np.abs((y_true - preds_mean) / np.where(y_true == 0, 1, y_true))) * 100
        
        print(f"   [MEAN] SỐ LIỆU TẬP TEST ({timeframe}):")
        print(f"   - RMSE: {rmse:.4f}")
        print(f"   - MAE:  {mae:.4f}")
        print(f"   - MAPE: {mape:.2f}%")
        
        # Lưu kết quả MEAN ra file CSV
        results_mean = pd.DataFrame({
            'timestamp': df_test['timestamp'].values,
            target_col: y_true,
            'predicted': preds_mean
        })
        
        path_mean = f"{output_dir}/results_xgboost_test_{target_col}_{timeframe}.csv"
        results_mean.to_csv(path_mean, index=False)
        print(f"   ✅ Đã lưu file MEAN tại: {path_mean}")
    else:
        print(f"❌ Không tìm thấy model MEAN tại: {model_path}")

    # ==============================================================================
    # PHẦN 2: XỬ LÝ MODEL Q90 (Chỉ chạy nếu file model tồn tại)
    # ==============================================================================
    # Tạo đường dẫn model Q90 dựa trên đường dẫn model chính
    # Ví dụ: models/xgboost/xgboost_y_req_t1_1m.pkl -> models/xgboost/xgboost_y_req_t1_1m_q90.pkl
    q90_model_path = model_path.replace(".pkl", "_q90.pkl")
    
    if os.path.exists(q90_model_path):
        print(f"🔹 Đang chạy model Q90...")
        model_q90 = joblib.load(q90_model_path)
        
        # Dự báo Q90
        preds_q90_log = model_q90.predict(X_test)
        preds_q90 = np.expm1(preds_q90_log)
        preds_q90 = np.maximum(preds_q90, 0)
        
        # Lưu kết quả Q90 ra file CSV riêng biệt
        results_q90 = pd.DataFrame({
            'timestamp': df_test['timestamp'].values,
            target_col: y_true,
            'predicted': preds_q90
        })
        
        path_q90 = f"{output_dir}/results_xgboost_test_{target_col}_{timeframe}_q90.csv"
        results_q90.to_csv(path_q90, index=False)
        print(f"   ✅ Đã lưu file Q90 tại: {path_q90}")
    else:
        # Q90 là tùy chọn, không báo lỗi đỏ, chỉ thông báo
        print(f"ℹ️ Không tìm thấy model Q90 (bỏ qua): {q90_model_path}")

if __name__ == "__main__":
    # Cấu hình danh sách cần chạy
    timeframes = ['1m', '5m', '15m']
    targets = ['y_req_t1', 'y_bytes_imp_t1']
    
    for tf in timeframes:
        for target in targets:
            # Đường dẫn file model Mean (Gốc)
            MODEL_FILE = f"models/xgboost/xgboost_{target}_{tf}.pkl"
            
            try:
                run_test_inference(
                    model_path=MODEL_FILE, 
                    timeframe=tf, 
                    target_col=target
                )
            except Exception as e:
                print(f"❌ Lỗi khi xử lý {target} {tf}: {e}")