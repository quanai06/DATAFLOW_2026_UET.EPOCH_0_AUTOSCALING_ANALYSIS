import numpy as np
import pandas as pd
from optimizer import AutoscalingOptimizer



# 1. ĐỌC DỮ LIỆU THẬT (Thay vì dùng np.random)
# Bạn tìm đường dẫn đến file parquet trong folder features của bạn
# df_features = pd.read_parquet('../../data/features/07_bytes_train.parquet') 

# Giả sử bạn lấy cột 'bytes' làm tải thực tế
# Bạn nên lấy khoảng 200 dòng đầu để test cho nhanh
# actual_load = df_features['bytes'].head(200).values
# forecast_load = df_features['bytes'].head(200).values

# BƯỚC 1: TẠO DỮ LIỆU GIẢ (Vì chưa có model)
# Giả sử chúng ta theo dõi trong 100 phút
minutes = 100
# Tạo dữ liệu tải thực tế: lúc thấp lúc cao (từ 200 đến 1800 đơn vị)
actual_load = np.random.randint(200, 2000, size=minutes)
# Giả vờ đây là kết quả model dự báo (lệch một chút so với thực tế)
forecast_load = actual_load * (1 + np.random.uniform(-0.1, 0.1, size=minutes))

# BƯỚC 2: CẤU HÌNH HỆ THỐNG
# Giả sử 1 server chịu được tối đa 1000 đơn vị tải
CAPACITY = 1000 
opt = AutoscalingOptimizer(server_capacity=CAPACITY, cooldown_minutes=5)

# BƯỚC 3: CHẠY THỬ
results = []
for t in range(minutes):
    # Đưa chuỗi dự báo vào (cần lấy cả đoạn từ đầu đến t để check luật 5 phút)
    current_servers = opt.check_scaling(forecast_load[:t+1], t)
    
    results.append({
        'Phút': t,
        'Tải_Thực': actual_load[t],
        'Dự_Báo': forecast_load[t],
        'Số_Server': current_servers
    })

# BƯỚC 4: XEM KẾT QUẢ
df = pd.DataFrame(results)
print(df.to_string())


# Giả định các thông số kinh tế
UNIT_COST_PER_MINUTE = 100  # 100 đồng mỗi phút cho 1 server
CAPACITY = 1000             # Sức tải đã đặt ở trên

# 1. Chi phí khi dùng Autoscaling (Của bạn)
# Tổng số server chạy qua mỗi phút nhân với đơn giá
total_cost_autoscaling = df['Số_Server'].sum() * UNIT_COST_PER_MINUTE

# 2. Chi phí khi dùng Cố định (Baseline - Luôn bật mức tối đa để an toàn)
# Giả sử mức tối đa là 4 server (như kết quả log của bạn)
max_servers = df['Số_Server'].max()
total_cost_fixed = len(df) * max_servers * UNIT_COST_PER_MINUTE

# 3. Tính toán Hiệu năng (Hệ thống có bị quá tải không?)
# Quá tải khi: Tải thực tế > Tổng sức tải của dàn máy đang bật
df['Overload'] = df['Tải_Thực'] > (df['Số_Server'] * CAPACITY)
total_overload_minutes = df['Overload'].sum()

# --- XUẤT BÁO CÁO ---
print("\n" + "="*30)
print("BÁO CÁO TỐI ƯU CHI PHÍ")
print("="*30)
print(f"1. Tổng chi phí Autoscaling: {total_cost_autoscaling:,} VNĐ")
print(f"2. Tổng chi phí Cố định (Fixed): {total_cost_fixed:,} VNĐ")
print(f"3. Số tiền tiết kiệm được: {total_cost_fixed - total_cost_autoscaling:,} VNĐ")
print(f"4. Tỷ lệ tiết kiệm: {((total_cost_fixed - total_cost_autoscaling)/total_cost_fixed)*100:.2f}%")
print(f"5. Số phút bị quá tải (SLA Violation): {total_overload_minutes} phút")
print(f"6. Tỷ lệ ổn định: {(1 - total_overload_minutes/len(df))*100:.2f}%")
print("="*30)