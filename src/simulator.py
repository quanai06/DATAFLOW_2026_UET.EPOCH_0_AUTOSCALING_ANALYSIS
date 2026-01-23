import pandas as pd
from src.optimizer import AutoscalingOptimizer

def run_simulation(actual_data, forecast_data, capacity, cost_per_server):
    # Khởi tạo bộ não Optimizer
    optimizer = AutoscalingOptimizer(server_capacity=capacity)
    
    server_history = []
    total_cost = 0
    overload_events = 0
    
    # Chạy mô phỏng qua từng phút (giả sử dữ liệu là khung 1m)
    for i in range(len(actual_data)):
        # 1. Lấy quyết định số server dựa trên dự báo
        # forecast_data[:i+1] giúp lấy chuỗi dự báo tính đến thời điểm hiện tại
        num_servers = optimizer.check_scaling(forecast_data[:i+1], i)
        server_history.append(num_servers)
        
        # 2. Tính tiền (đơn giản hóa: số server * đơn giá mỗi phút) 
        total_cost += num_servers * (cost_per_server / 60)
        
        # 3. Kiểm tra xem có bị "sập" không (Tải thực > Tổng sức tải dàn máy) [cite: 7]
        if actual_data[i] > (num_servers * capacity):
            overload_count += 1
            
    return server_history, total_cost, overload_events