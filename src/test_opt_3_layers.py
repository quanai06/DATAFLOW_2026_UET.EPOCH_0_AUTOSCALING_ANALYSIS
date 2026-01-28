import pandas as pd
import optuna
import os
from optimize_3_layers import HybridOptimizer 

# --- BƯỚC 1: LOAD DỮ LIỆU ĐÃ MERGE (1m, 5m, 15m) ---
# Hàm này nạp file CSV kết quả dự báo mà bạn đã chạy từ XGBoost và hợp nhất chúng lại
# Chuyển về dạng req/s và bytes/s cho từng khoảng thời gian
# Xử lý khoảng trống bằng forward-fill

def change_to_per_second(model, timeframe, suffix='' ):
    df_req = pd.read_csv(os.path.join('results', model, f'results_{model}_y_req_t1_{timeframe}{suffix}.csv'))
    df_bytes = pd.read_csv(os.path.join('results', model, f'results_{model}_y_bytes_imp_t1_{timeframe}{suffix}.csv'))
    
    # Chuyển đổi dự báo về đơn vị per second
    factor = {'1m': 60, '5m': 300, '15m': 900}[timeframe]
    
    df_req['y_req_t1'] = df_req['y_req_t1'] / factor
    df_bytes['y_bytes_imp_t1'] = df_bytes['y_bytes_imp_t1'] / factor
    df_req['predicted'] = df_req['predicted'] / factor
    df_bytes['predicted'] = df_bytes['predicted'] / factor
    
    df_req = df_req.rename(columns={'y_req_t1': f'act_{timeframe}{suffix}_req', 'predicted': f'predicted_{timeframe}{suffix}_req'})
    df_bytes = df_bytes.rename(columns={'y_bytes_imp_t1': f'act_{timeframe}{suffix}_bytes', 'predicted': f'predicted_{timeframe}{suffix}_bytes'})
    df_req['timestamp'] = pd.to_datetime(df_req['timestamp'])
    df_bytes['timestamp'] = pd.to_datetime(df_bytes['timestamp'])
    return df_req, df_bytes

def merge_multiresolution_data(model):
    print(">>> Đang hợp nhất dữ liệu đa độ phân giải...")
    
    # Load và đổi đơn vị
    req_1m, bytes_1m = change_to_per_second(model, '1m')
    req_5m, bytes_5m = change_to_per_second(model, '5m')
    req_5m_q90, bytes_5m_q90 = change_to_per_second(model, '5m', suffix='_q90')
    req_15m, bytes_15m = change_to_per_second(model, '15m')
    
    # Merge
    df = pd.merge(req_1m, bytes_1m, on='timestamp')
    
    df = pd.merge(df, req_5m[['timestamp', 'predicted_5m_req']], on='timestamp', how='left')
    df = pd.merge(df, bytes_5m[['timestamp', 'predicted_5m_bytes']], on='timestamp', how='left')
    df = pd.merge(df, req_5m_q90[['timestamp', 'predicted_5m_q90_req']], on='timestamp', how='left')
    df = pd.merge(df, bytes_5m_q90[['timestamp', 'predicted_5m_q90_bytes']], on='timestamp', how='left')
    df = pd.merge(df, req_15m[['timestamp', 'predicted_15m_req']], on='timestamp', how='left')
    df = pd.merge(df, bytes_15m[['timestamp', 'predicted_15m_bytes']], on='timestamp', how='left')
    
    # Xử lý khoảng trống bằng forward-fill
    df = df.sort_values('timestamp')
    df = df.ffill()
    df = df.dropna().reset_index(drop=True)

    return df

# --- BƯỚC 2: ĐỊNH NGHĨA HÀM OBJECTIVE ---

def objective(trial, df):
    # Các tham số Optuna sẽ "thử sai"
    k_base = trial.suggest_float("k_base", 0.8, 1.2)
    alpha_5m = trial.suggest_float("alpha_5m", 0.0, 1.5)
    panic_threshold = trial.suggest_int("panic_threshold", 50, 200)
    burst_add = trial.suggest_int("burst_add", 1, 5)
    patience = trial.suggest_int("patience", 5, 15)

    # Khởi tạo bộ não tối ưu với bộ tham số trial này
    opt = HybridOptimizer(500, 4000, k_base, alpha_5m, panic_threshold, burst_add, patience)

    cost_run = 0
    sla_penalty = 0
    scaling_events = 0
    prev_s = 1

    # Chạy mô phỏng qua toàn bộ tập dữ liệu
    for t in range(len(df)):
        # Truyền dữ liệu vào tầng step
        s = opt.step(
            f15_req=df['predicted_15m_req'].iloc[t],
            f15_bytes=df['predicted_15m_bytes'].iloc[t],
            f5_req=df['predicted_5m_req'].iloc[t],
            f5_bytes=df['predicted_5m_bytes'].iloc[t],
            f5_req_q90=df['predicted_5m_q90_req'].iloc[t],
            f5_bytes_q90=df['predicted_5m_q90_bytes'].iloc[t],
            act1_req=df['act_1m_req'].iloc[t],
            act1_bytes=df['act_1m_bytes'].iloc[t],
            t=t
        )
        
        cost_run += s # Cộng dồn tiền thuê máy chủ
        
        # 1. Phạt vi phạm SLA (Thiếu hụt thực tế so với đáp ứng)
        # Check SLA theo công suất cực hạn
        if df['act_1m_req'].iloc[t] > s * 500 or df['act_1m_bytes'].iloc[t] > s * 4000:
            sla_penalty += 1 
            
        # 2. Phạt dao động (Scaling Events - Chống Flapping)
        if s != prev_s:
            scaling_events += 1
        prev_s = s
            
    # HÀM MỤC TIÊU: Kết hợp 3 loại chi phí
    # Có thể điều chỉnh trọng số (100, 5000, 50) tùy vào độ ưu tiên
    total_score = (cost_run * 100) + (sla_penalty * 5000) + (scaling_events * 500)
    return total_score

# --- BƯỚC 3: KÍCH HOẠT OPTUNA ---
def run_optimization(model):
    study = optuna.create_study(direction="minimize")

    df = merge_multiresolution_data(model)
    study.optimize(lambda trial: objective(trial, df), n_trials=100) # Thử 100 bộ tham số khác nhau

    print("=== KẾT QUẢ TỐI ƯU CHIẾN LƯỢC ===")
    print(f"Giá trị Cost thấp nhất: {study.best_value}")
    print(f"Bộ tham số tốt nhất: {study.best_params}")
    
    # Lưu bộ tham số này lại để dùng cho bản Demo/Báo cáo
    import json
    with open(f'results/{model}_best_strategy_params.json', 'w') as f:
        json.dump(study.best_params, f)

if __name__ == "__main__":
    run_optimization('xgboost')