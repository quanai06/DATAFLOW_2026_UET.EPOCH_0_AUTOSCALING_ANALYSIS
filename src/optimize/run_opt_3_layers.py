import pandas as pd
import optuna
import os
from src.optimize.universalOptimizer import HybridOptimizer
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

CAP_REQ = int(os.getenv('CAP_REQ'))
CAP_BYTES = int(os.getenv('CAP_BYTES'))
COST_PER_SERVER_PER_MIN = int(os.getenv('COST_PER_SERVER_PER_MIN'))
COST_SOFT_SLA_PENALTY_REQ = int(os.getenv('COST_SOFT_SLA_PENALTY_REQ'))
COST_SOFT_SLA_PENALTY_BYTES = int(os.getenv('COST_SOFT_SLA_PENALTY_BYTES'))
COST_HARD_SLA_PENALTY_REQ = int(os.getenv('COST_HARD_SLA_PENALTY_REQ'))
COST_HARD_SLA_PENALTY_BYTES = int(os.getenv('COST_HARD_SLA_PENALTY_BYTES'))
COST_SCALING_EVENT = int(os.getenv('COST_SCALING_EVENT'))
TARGET_UTIL = float(os.getenv('TARGET_UTIL'))
MAX_SERVERS = int(os.getenv('MAX_SERVERS'))

# ĐỊNH NGHĨA HÀM OBJECTIVE ---

def objective(trial, df):
    
    # Các tham số Optuna sẽ "thử sai"
    k_base = trial.suggest_float("k_base", 0.8, 1.2)
    alpha_5m = trial.suggest_float("alpha_5m", 0.0, 1.5)
    max_reduce = trial.suggest_int("max_reduce", 1, 5)
    panic_ratio = trial.suggest_float("panic_ratio", 0.05, 0.5)
    burst_add = trial.suggest_int("burst_add", 1, 5)
    patience = trial.suggest_int("patience", 5, 15)
    
    # Khởi tạo bộ não tối ưu với bộ tham số trial này
    opt = HybridOptimizer(CAP_REQ, CAP_BYTES, k_base, alpha_5m, max_reduce, panic_ratio, 
                          burst_add, patience, target_util=TARGET_UTIL, max_servers=MAX_SERVERS)

    count_servers = 0
    scaling_events = 0
    prev_s = 1
    total_soft_overload_req = 0
    total_soft_overload_bytes = 0
    total_hard_overload_req = 0
    total_hard_overload_bytes = 0

    # Chạy mô phỏng qua toàn bộ tập dữ liệu
    for t, row in enumerate(df.itertuples()):
        t = row.Index # Giả sử index đã reset 0..N
        
        # Bước step của thuật toán
        s = opt.step(
            f15_req=row.predicted_15m_req,
            f15_bytes=row.predicted_15m_bytes,
            f5_req=row.predicted_5m_req,
            f5_req_q90=row.predicted_5m_q90_req,
            f5_bytes=row.predicted_5m_bytes,
            f5_bytes_q90=row.predicted_5m_q90_bytes,
            act1_req=row.act_1m_req,
            act1_bytes=row.act_1m_bytes,
            t=t
        )
        count_servers += s 

        # Tính shortage (thiếu hụt) tại thời điểm t
        soft_overload_req = max(0, min(row.act_1m_req, s * CAP_REQ) - s * CAP_REQ * opt.target_util)
        soft_overload_bytes = max(0, min(row.act_1m_bytes, s * CAP_BYTES) - s * CAP_BYTES * opt.target_util)

        hard_overload_req = max(0, row.act_1m_req - s * CAP_REQ)
        hard_overload_bytes = max(0, row.act_1m_bytes - s * CAP_BYTES)
        
        total_soft_overload_req += soft_overload_req
        total_soft_overload_bytes += soft_overload_bytes 
        total_hard_overload_req += hard_overload_req
        total_hard_overload_bytes += hard_overload_bytes
        
        # Phạt dao động (Scaling Events - Chống Flapping)
        if s != prev_s:
            scaling_events += abs(s - prev_s)
        prev_s = s
    
    sla_penalty = total_soft_overload_req * COST_SOFT_SLA_PENALTY_REQ + \
                        total_soft_overload_bytes * COST_SOFT_SLA_PENALTY_BYTES + \
                        total_hard_overload_req * COST_HARD_SLA_PENALTY_REQ + \
                        total_hard_overload_bytes * COST_HARD_SLA_PENALTY_BYTES
    
    # HÀM MỤC TIÊU MỚI: Ưu tiên giảm thiểu diện tích thiếu hụt
    # Trọng số của shortage cần cao để AI "sợ" sập nặng
    total_score = (count_servers * COST_PER_SERVER_PER_MIN) + sla_penalty + \
                (scaling_events * COST_SCALING_EVENT)

    return total_score

# --- BƯỚC 3: KÍCH HOẠT OPTUNA ---
def run_optimization(model):
    study = optuna.create_study(direction="minimize")

    df = pd.read_csv(f'results/merged_{model}_data.csv')

    study.optimize(lambda trial: objective(trial, df), n_trials=100) # Thử 100 bộ tham số khác nhau

    print("=== KẾT QUẢ TỐI ƯU CHIẾN LƯỢC ===")
    print(f"Giá trị Cost thấp nhất: {study.best_value}")
    print(f"Bộ tham số tốt nhất: {study.best_params}")
    
    # Lưu bộ tham số này và giá trị tối ưu nhất lại để dùng cho bản Demo/Báo cáo
    import json
    with open(f'results/{model}/{model}_best_strategy_params.json', 'w') as f:
        json.dump({"best_params": study.best_params, "best_value": study.best_value}, f)

    # Lưu data sau khi merge để dùng cho bản Demo/Báo cáo
    df.to_csv(f'results/{model}/{model}_merged_multiresolution_data.csv', index=False)
    print(f"Đã lưu dữ liệu hợp nhất tại: results/{model}/{model}_merged_multiresolution_data.csv")

    # Lưu các giá trị vào file txt để tiện theo dõi
    with open(f'results/{model}/{model}_optimization_summary.txt', 'w', encoding='utf-8') as f:
        f.write(f"KẾT QUẢ TỐI ƯU CHIẾN LƯỢC\n")
        f.write(f"Giá trị Cost thấp nhất: {study.best_value}\n")
        f.write(f"Bộ tham số tốt nhất: {study.best_params}\n")


if __name__ == "__main__":
    run_optimization('xgboost')
    # run_optimization('lgbm')