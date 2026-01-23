import numpy as np
import pandas as pd
import math
from optimizer import AutoscalingOptimizer

# # --- BƯỚC 1: NẠP DỮ LIỆU THỰC TẾ TỪ KẾT QUẢ MODEL ---

# # File này chứa kết quả dự báo trên tập Test (từ ngày 23/8 đến 31/8)
# df_results_req = pd.read_csv('req_model_results.csv') 
# df_results_bytes = pd.read_csv('bytes_model_results.csv') 

# # Nạp các chuỗi dữ liệu
# actual_req = df_results_req['actual_requests'].values
# forecast_req = df_results_req['predicted_requests'].values

# actual_bytes = df_results_bytes['actual_bytes'].values
# forecast_bytes = df_results_bytes['predicted_bytes'].values
# minutes = len(df_results_req) # Số phút chạy mô phỏng dựa trên độ dài tập Test

# --- BƯỚC 1: TẠO DỮ LIỆU GIẢ LẬP ĐA BIẾN ---
minutes = 100
np.random.seed(42) # Giữ kết quả cố định để dễ theo dõi

# Tạo tải Request: Dao động bình thường
actual_req = np.random.randint(200, 1500, size=minutes)
forecast_req = actual_req * (1 + np.random.uniform(-0.1, 0.1, size=minutes))

# Tạo tải Bytes: Thỉnh thoảng có Spike cực lớn (Outliers)
actual_bytes = np.random.randint(1000, 5000, size=minutes)
actual_bytes[40:45] = 15000  # Tạo một đợt Spike dung lượng lớn ở phút 40
forecast_bytes = actual_bytes * (1 + np.random.uniform(-0.1, 0.1, size=minutes))

# --- BƯỚC 2: CẤU HÌNH HỆ THỐNG ---
CAP_REQ = 1000    # 1 server chịu được 1000 req/phút
CAP_BYTES = 8000  # 1 server chịu được 8000 bytes/phút
UNIT_COST_PER_MINUTE = 100 
PENALTY_PER_MINUTE = 1000 # Phạt nặng nếu overload

opt = AutoscalingOptimizer(capacity_req=CAP_REQ, capacity_bytes=CAP_BYTES, cooldown_minutes=5)

# --- BƯỚC 3: CHẠY MÔ PHỎNG ---
results = []
for t in range(minutes):
    # Truyền cả 2 chuỗi dự báo vào
    curr_serv = opt.check_scaling(forecast_req[:t+1], forecast_bytes[:t+1], t)
    
    results.append({
        'Minute': t,
        'Actual_Req': actual_req[t],
        'Actual_Bytes': actual_bytes[t],
        'Servers': curr_serv
    })

df = pd.DataFrame(results)

# --- BƯỚC 4: TÍNH TOÁN KINH TẾ & HIỆU NĂNG ---
# Check Overload: Nếu nghẽn 1 trong 2 cái là sập
df['Overload'] = (df['Actual_Req'] > df['Servers'] * CAP_REQ) | \
                 (df['Actual_Bytes'] > df['Servers'] * CAP_BYTES)

total_overload_mins = df['Overload'].sum()

# Chi phí Autoscaling = Tiền thuê + Tiền phạt sập web
cost_rental = df['Servers'].sum() * UNIT_COST_PER_MINUTE
cost_penalty = total_overload_mins * PENALTY_PER_MINUTE
total_cost_auto = cost_rental + cost_penalty

# Chi phí Cố định (Fixed): Luôn bật mức tối đa để không bao giờ sập
max_req_needed = math.ceil(df['Actual_Req'].max() / CAP_REQ)
max_bytes_needed = math.ceil(df['Actual_Bytes'].max() / CAP_BYTES)
fixed_servers = df['Servers'].max()
total_cost_fixed = len(df) * fixed_servers * UNIT_COST_PER_MINUTE

# --- XUẤT BÁO CÁO ---
print("\n" + "="*40)
print("BÁO CÁO TỐI ƯU CHI PHÍ ĐA TÀI NGUYÊN")
print("="*40)
print(f"Tổng số phút mô phỏng: {minutes}")
print(f"Số máy chủ tối đa đã dùng: {df['Servers'].max()}")
print(f"Số máy chủ nếu dùng cố định: {fixed_servers}")
print("-" * 40)
print(f"1. Chi phí Autoscaling: {total_cost_auto:,} VNĐ")
print(f"   (Tiền thuê: {cost_rental:,} | Tiền phạt: {cost_penalty:,})")
print(f"2. Chi phí Cố định:     {total_cost_fixed:,} VNĐ")
print(f"3. Tiết kiệm được:      {total_cost_fixed - total_cost_auto:,} VNĐ")
print(f"4. Tỷ lệ tiết kiệm:     {((total_cost_fixed - total_cost_auto)/total_cost_fixed)*100:.2f}%")
print(f"5. Số phút quá tải:     {total_overload_mins} phút")
print(f"6. Tỷ lệ ổn định:       {(1 - total_overload_mins/len(df))*100:.2f}%")
print("="*40)