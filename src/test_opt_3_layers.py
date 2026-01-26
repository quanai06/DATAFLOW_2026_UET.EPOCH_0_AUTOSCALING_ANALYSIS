import pandas as pd
import optuna
import os
from optimize_3_layers import HybridOptimizer 

# --- BƯỚC 1: LOAD DỮ LIỆU ĐÃ MERGE (1m, 5m, 15m) ---
# Hàm này nạp file CSV kết quả dự báo mà bạn đã chạy từ XGBoost và hợp nhất chúng lại

def merge_multiresolution_data(base_path='results/xgboost/'):
    print(">>> Đang hợp nhất dữ liệu đa độ phân giải...")
    
    # 1. Load dữ liệu 1 phút (Làm khung xương chính cho mọi phút)
    req_1m = pd.read_csv(os.path.join(base_path, 'results_xgb_y_req_t1_1m.csv'))
    req_1m = req_1m.rename(columns={'y_req_t1': 'act_req', 'predicted': 'f1m_req'})
    
    bytes_1m = pd.read_csv(os.path.join(base_path, 'results_xgb_y_bytes_imp_t1_1m.csv'))
    bytes_1m = bytes_1m.rename(columns={'y_bytes_imp_t1': 'act_bytes', 'predicted': 'f1m_bytes'})
    
    # Merge 1m Requests và Bytes
    df = pd.merge(req_1m, bytes_1m, on='timestamp')

    # 2. Load và gộp dữ liệu 5 phút
    req_5m = pd.read_csv(os.path.join(base_path, 'results_xgb_y_req_t1_5m.csv'))[['timestamp', 'predicted']]
    req_5m = req_5m.rename(columns={'predicted': 'f5m_req'})
    
    bytes_5m = pd.read_csv(os.path.join(base_path, 'results_xgb_y_bytes_imp_t1_5m.csv'))[['timestamp', 'predicted']]
    bytes_5m = bytes_5m.rename(columns={'predicted': 'f5m_bytes'})
    
    df = pd.merge(df, req_5m, on='timestamp', how='left')
    df = pd.merge(df, bytes_5m, on='timestamp', how='left')

    # 3. Load và gộp dữ liệu 15 phút
    req_15m = pd.read_csv(os.path.join(base_path, 'results_xgb_y_req_t1_15m.csv'))[['timestamp', 'predicted']]
    req_15m = req_15m.rename(columns={'predicted': 'f15m_req'})
    
    bytes_15m = pd.read_csv(os.path.join(base_path, 'results_xgb_y_bytes_imp_t1_15m.csv'))[['timestamp', 'predicted']]
    bytes_15m = bytes_15m.rename(columns={'predicted': 'f15m_bytes'})
    
    df = pd.merge(df, req_15m, on='timestamp', how='left')
    df = pd.merge(df, bytes_15m, on='timestamp', how='left')
    
    return df


# --- BƯỚC 2: ĐỊNH NGHĨA HÀM OBJECTIVE ---
def objective(trial, df=merge_multiresolution_data()):
    # Các tham số Optuna sẽ "thử sai"
    k_base = trial.suggest_float("k_base", 0.8, 1.2)
    buffer_5m = trial.suggest_float("buffer_5m", 0.1, 0.4)
    panic_threshold = trial.suggest_int("panic_threshold", 50, 200)
    burst_add = trial.suggest_int("burst_add", 1, 5)
    patience = trial.suggest_int("patience", 5, 15)

    # Khởi tạo bộ não tối ưu với bộ tham số trial này
    opt = HybridOptimizer(500, 4000, k_base, buffer_5m, panic_threshold, burst_add, patience)
    
    cost_run = 0
    sla_penalty = 0
    scaling_events = 0
    prev_s = 1

    # Chạy mô phỏng qua toàn bộ tập dữ liệu
    for t in range(len(df)):
        # Truyền dữ liệu vào tầng step
        s = opt.step(
            f15_req=df['f15m_req'].iloc[t],
            f15_bytes=df['f15m_bytes'].iloc[t],
            f5_req=df['f5m_req'].iloc[t],
            f5_bytes=df['f5m_bytes'].iloc[t],
            act1_req=df['act_req'].iloc[t],
            t=t
        )
        
        cost_run += s # Cộng dồn tiền thuê máy chủ
        
        # 1. Phạt vi phạm SLA (Thiếu hụt thực tế so với đáp ứng)
        if df['act_req'].iloc[t] > s * 500 or df['act_bytes'].iloc[t] > s * 4000:
            sla_penalty += 1 
            
        # 2. Phạt dao động (Scaling Events - Chống Flapping)
        if s != prev_s:
            scaling_events += 1
        prev_s = s
            
    # HÀM MỤC TIÊU: Kết hợp 3 loại chi phí
    # Có thể điều chỉnh trọng số (1000, 50) tùy vào độ ưu tiên
    total_score = (cost_run * 100) + (sla_penalty * 5000) + (scaling_events * 500)
    return total_score

# --- BƯỚC 3: KÍCH HOẠT OPTUNA ---
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100) # Thử 100 bộ tham số khác nhau

    print("=== KẾT QUẢ TỐI ƯU CHIẾN LƯỢC ===")
    print(f"Giá trị Cost thấp nhất: {study.best_value}")
    print(f"Bộ tham số tốt nhất: {study.best_params}")
    
    # Lưu bộ tham số này lại để dùng cho bản Demo/Báo cáo
    import json
    with open('results/best_strategy_params.json', 'w') as f:
        json.dump(study.best_params, f)