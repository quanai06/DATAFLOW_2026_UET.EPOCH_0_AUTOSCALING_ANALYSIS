import numpy as np
import pandas as pd
import math
import os
from optimizer import AutoscalingOptimizer

# --- BƯỚC 1: NẠP VÀ HỢP NHẤT DỮ LIỆU THỰC TẾ TỪ MODEL ---

base_path = 'results/xgboost/'

# Đọc kết quả khung 1 phút (nhạy bén nhất cho Autoscaling)
df_req = pd.read_csv(os.path.join(base_path, 'results_xgb_y_req_t1_1m.csv'))
df_bytes = pd.read_csv(os.path.join(base_path, 'results_xgb_y_bytes_imp_t1_1m.csv'))

# Đổi tên để tránh trùng lặp khi merge
df_req = df_req.rename(columns={'y_req_t1': 'actual_req', 'predicted': 'forecast_req'})
df_bytes = df_bytes.rename(columns={'y_bytes_imp_t1': 'actual_bytes', 'predicted': 'forecast_bytes'})

# Gộp dữ liệu theo timestamp
df_final = pd.merge(df_req, df_bytes, on='timestamp').sort_values('timestamp')

# Chuyển thành numpy array để chạy simulation
actual_req = df_final['actual_req'].values
forecast_req = df_final['forecast_req'].values
actual_bytes = df_final['actual_bytes'].values
forecast_bytes = df_final['forecast_bytes'].values
timestamps = df_final['timestamp'].values

minutes = len(df_final)

# --- BƯỚC 2: CẤU HÌNH HỆ THỐNG DỰA TRÊN THỰC TẾ DATA ---

# Tìm Max Load để đặt Capacity hợp lý
max_r = actual_req.max()
max_b = actual_bytes.max()

# Giả định hệ thống chịu tải đỉnh bằng khoảng 4 máy chủ
CAP_REQ = math.ceil(max_r / 3.5)   # Ví dụ: 1500 / 3.5 ≈ 430
CAP_BYTES = math.ceil(max_b / 3.5) # Ví dụ: 6000 / 3.5 ≈ 1700

UNIT_COST_PER_MINUTE = 100 
PENALTY_PER_MINUTE = 1000 # Phạt nặng để ưu tiên độ ổn định

# Khởi tạo Optimizer với target_util = 0.8 (chạy 80% công suất cho an toàn)
opt = AutoscalingOptimizer(
    capacity_req=CAP_REQ, 
    capacity_bytes=CAP_BYTES, 
    cooldown_minutes=5, 
    target_util=0.3
)

# --- BƯỚC 3: CHẠY MÔ PHỎNG ---
results = []
for t in range(minutes):
    # Optimizer nhận chuỗi dự báo t+1
    current_servers = opt.check_scaling(forecast_req[:t+1], forecast_bytes[:t+1], t)
    
    results.append({
        'Phút': t,
        'timestamp': timestamps[t],
        'Actual_Req': actual_req[t],
        'Actual_Bytes': actual_bytes[t],
        'Predicted_Req': forecast_req[t],
        'Predicted_Bytes': forecast_bytes[t],
        'Số_Server': current_servers
    })

df_res = pd.DataFrame(results)

# --- BƯỚC 4: TÍNH TOÁN KINH TẾ & HIỆU NĂNG ---

# Kiểm tra Overload thực tế (Nghẽn Req hoặc Nghẽn Bytes)
df_res['Overload'] = (df_res['Actual_Req'] > df_res['Số_Server'] * CAP_REQ) | \
                     (df_res['Actual_Bytes'] > df_res['Số_Server'] * CAP_BYTES)

total_overload_minutes = df_res['Overload'].sum()

# Chi phí thực tế của giải pháp Autoscaling
total_cost_autoscaling = (df_res['Số_Server'].sum() * UNIT_COST_PER_MINUTE) + \
                         (total_overload_minutes * PENALTY_PER_MINUTE)

# Chi phí Cố định (Luôn thuê mức cao nhất dàn máy từng chạm tới)
fixed_servers = df_res['Số_Server'].max()
total_cost_fixed = len(df_res) * fixed_servers * UNIT_COST_PER_MINUTE

# --- XUẤT BÁO CÁO ---
print("\n" + "="*45)
print("BÁO CÁO TỐI ƯU CHI PHÍ DỰA TRÊN MODEL XGBOOST")
print("="*45)
print(f"Tổng thời gian theo dõi: {minutes} phút")
print(f"Sức tải thiết lập: {CAP_REQ} req/m | {CAP_BYTES} bytes/m")
print(f"Số máy chủ tối đa sử dụng: {fixed_servers}")
print("-" * 45)
print(f"1. Tổng chi phí Autoscaling: {total_cost_autoscaling:,} VNĐ")
print(f"2. Tổng chi phí Cố định (Fixed): {total_cost_fixed:,} VNĐ")
print(f"3. Số tiền tiết kiệm được: {total_cost_fixed - total_cost_autoscaling:,} VNĐ")
print(f"4. Tỷ lệ tiết kiệm: {((total_cost_fixed - total_cost_autoscaling)/total_cost_fixed)*100:.2f}%")
print(f"5. Số phút bị quá tải (SLA): {total_overload_minutes} phút")
print(f"6. Tỷ lệ ổn định hệ thống: {(1 - total_overload_minutes/len(df_res))*100:.2f}%")
print("="*45)

# Lưu lại kết quả để vẽ biểu đồ cho báo cáo
df_res.to_csv('results/xgboost/final_optimization_results.csv', index=False)