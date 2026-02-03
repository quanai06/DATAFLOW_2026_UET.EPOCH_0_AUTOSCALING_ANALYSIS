import pandas as pd
import optuna
import os
from universalOptimizer import UniversalOptimizer
from dotenv import load_dotenv
import json

# Load environment variables from .env file
load_dotenv()

CAP_REQ = float(os.getenv('CAP_REQ'))
CAP_BYTES = float(os.getenv('CAP_BYTES'))
COST_PER_SERVER_PER_MIN = int(os.getenv('COST_PER_SERVER_PER_MIN'))
COST_SOFT_SLA_PENALTY_REQ = int(os.getenv('COST_SOFT_SLA_PENALTY_REQ'))
COST_SOFT_SLA_PENALTY_BYTES = int(os.getenv('COST_SOFT_SLA_PENALTY_BYTES'))
COST_HARD_SLA_PENALTY_REQ = int(os.getenv('COST_HARD_SLA_PENALTY_REQ'))
COST_HARD_SLA_PENALTY_BYTES = int(os.getenv('COST_HARD_SLA_PENALTY_BYTES'))
COST_SCALING_EVENT = int(os.getenv('COST_SCALING_EVENT'))
TARGET_UTIL = float(os.getenv('TARGET_UTIL'))
MAX_SERVERS = int(os.getenv('MAX_SERVERS'))

eff_cap_req = CAP_REQ * TARGET_UTIL
eff_cap_bytes = CAP_BYTES * TARGET_UTIL

# ĐỊNH NGHĨA HÀM OBJECTIVE ---

def objective(trial, df, strategy):

    # Định nghĩa tham số riêng cho từng chiến lược để Optuna thử nghiệm
    if strategy == "reactive":
        params = {
            "max_reduce": trial.suggest_int("max_reduce_r", 1, 3),
            "in_patience": trial.suggest_int("in_patience_r", 5, 15),
            "out_patience": trial.suggest_int("out_patience_r", 3, 5)
        }
    elif strategy == "predictive":
        params = {
            "k_base": trial.suggest_float("k_base_p", 0.8, 1.2),
            "alpha_5m": trial.suggest_float("alpha_5m_p", 0.2, 1),
            "in_patience": trial.suggest_int("in_patience_p", 5, 15)

        }
    else: # hybrid
        params = {
            "k_base": trial.suggest_float("k_base_h", 0.8, 1.2),
            "alpha_5m": trial.suggest_float("alpha_5m_h", 0.2, 1.0),
            "panic_ratio": trial.suggest_float("panic_ratio_h", 0.05, 0.5),
            "burst_add": trial.suggest_int("burst_add_h", 1, 5),
            "in_patience": trial.suggest_int("in_patience_h", 10, 15),
            "max_reduce": trial.suggest_int("max_reduce_h", 1, 2)
        }

    # Khởi tạo bộ não tối ưu với bộ tham số trial này
    opt = UniversalOptimizer (mode=strategy, cap_req=CAP_REQ, cap_bytes=CAP_BYTES, 
                              target_util=TARGET_UTIL, max_servers=MAX_SERVERS, **params)

    count_servers = 0
    scaling_events = 0
    prev_s = 1
    total_soft_overload_req = 0
    total_soft_overload_bytes = 0
    total_hard_overload_req = 0
    total_hard_overload_bytes = 0

    # Chạy mô phỏng qua toàn bộ tập dữ liệu
    for t, row in enumerate(df.itertuples()):
        
        # Bước step của thuật toán
        s = opt.step(row, t)

        count_servers += s 

        # Tính shortage (thiếu hụt) tại thời điểm t
        soft_overload_req = max(0, min(row.act_1m_req, s * CAP_REQ) - s * eff_cap_req)
        soft_overload_bytes = max(0, min(row.act_1m_bytes, s * CAP_BYTES) - s * eff_cap_bytes)

        hard_overload_req = max(0, row.act_1m_req - s * CAP_REQ)
        hard_overload_bytes = max(0, row.act_1m_bytes - s * CAP_BYTES)
        
        total_soft_overload_req += soft_overload_req
        total_soft_overload_bytes += soft_overload_bytes 
        total_hard_overload_req += hard_overload_req
        total_hard_overload_bytes += hard_overload_bytes
        
        # Phạt dao động (Scaling Events - Chống Flapping)
        if s != prev_s:
            scaling_events += 1 # mỗi lần thay đổi số server
            # scaling_events += abs(s - prev_s)
        prev_s = s
    
    sla_penalty = total_soft_overload_req * COST_SOFT_SLA_PENALTY_REQ + \
                        total_soft_overload_bytes * COST_SOFT_SLA_PENALTY_BYTES + \
                        total_hard_overload_req * COST_HARD_SLA_PENALTY_REQ + \
                        total_hard_overload_bytes * COST_HARD_SLA_PENALTY_BYTES
    
    # HÀM MỤC TIÊU MỚI: Ưu tiên giảm thiểu diện tích thiếu hụt
    # Trọng số của shortage cần cao để AI "sợ" sập nặng
    total_score = (count_servers * COST_PER_SERVER_PER_MIN) + sla_penalty + \
                (scaling_events * COST_SCALING_EVENT)

    trial.set_user_attr("scaling_events", scaling_events)
    trial.set_user_attr("total_soft_overload",
                        total_soft_overload_req + total_soft_overload_bytes)
    trial.set_user_attr("total_hard_overload",
                        total_hard_overload_req + total_hard_overload_bytes)


    return total_score

# --- BƯỚC 3: KÍCH HOẠT OPTUNA ---
def run_optimization(model, strategy):
    study = optuna.create_study(direction="minimize")

    df = pd.read_csv(f'results/merged_train_data/merged_{model}_data.csv')

    study.optimize(lambda trial: objective(trial, df, strategy), n_trials=100) # Thử 100 bộ tham số khác nhau

    print("=== KẾT QUẢ TỐI ƯU CHIẾN LƯỢC ===")
    print(f"Giá trị Cost thấp nhất: {study.best_value}")
    print(f"Bộ tham số tốt nhất: {study.best_params}")
    
    # Lưu bộ tham số này và giá trị tối ưu nhất lại để dùng cho bản Demo/Báo cáo
    path_json=f'results/optimize_train_data/{model}/{model}_{strategy}_best_strategy_params.json'
    os.makedirs(os.path.dirname(path_json), exist_ok=True)
    with open(path_json, 'w') as f:
        json.dump({"strategy": strategy, "best_params": study.best_params, "best_value": study.best_value}, f)

    # Lưu các giá trị vào file txt để tiện theo dõi
    path_txt=f'results/optimize_train_data/{model}/{model}_{strategy}_optimization_summary.txt'
    os.makedirs(os.path.dirname(path_txt), exist_ok=True)
    with open(path_txt, 'w', encoding='utf-8') as f:
        f.write(f"KẾT QUẢ TỐI ƯU CHIẾN LƯỢC\n")
        f.write(f"Chiến lược: {strategy}\n")
        f.write(f"Giá trị Cost thấp nhất: {study.best_value}\n")
        f.write(f"Bộ tham số tốt nhất: {study.best_params}\n")
        f.write(f"Số lần scaling events trong mô phỏng: {study.best_trial.user_attrs.get('scaling_events', 'N/A')}\n")
        f.write(f"Tổng số lần soft overload (req + bytes): {study.best_trial.user_attrs.get('total_soft_overload', 'N/A')}\n")
        f.write(f"Tổng số lần hard overload (req + bytes): {study.best_trial.user_attrs.get('total_hard_overload', 'N/A')}\n")
        f.write("\n")

if __name__ == "__main__":
    run_optimization('xgboost', 'reactive')
    run_optimization('xgboost', 'predictive')
    run_optimization('xgboost', 'hybrid')
    
    run_optimization('lgbm', 'reactive')
    run_optimization('lgbm', 'predictive')
    run_optimization('lgbm', 'hybrid')

    run_optimization('lstm', 'reactive')
    run_optimization('lstm', 'predictive')
    run_optimization('lstm', 'hybrid')
