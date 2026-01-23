import pandas as pd
import numpy as np

class AutoscalingOptimizer:
    def __init__(self, server_capacity, cooldown_minutes=5, threshold_ratio=0.8):
        self.server_capacity = server_capacity # Sức tải mỗi server (req/phút)
        self.cooldown_period = cooldown_minutes # Cooldown tránh flapping 
        self.threshold = server_capacity * threshold_ratio # Ngưỡng để scale-out
        self.current_servers = 1
        self.last_scale_time = -cooldown_minutes # Đảm bảo có thể scale ngay lúc đầu
        self.decision_history = []

    def check_scaling(self, forecast_series, current_time):
        """
        Logic: Scale-out khi dự báo > ngưỡng trong 5 phút liên tiếp 
        """
        # Kiểm tra nếu đang trong thời gian cooldown
        if current_time - self.last_scale_time < self.cooldown_period:
            return self.current_servers

        # Lấy 5 phút dự báo gần nhất
        recent_forecast = forecast_series[-5:] 
        
        # Nếu tất cả 5 phút đều vượt ngưỡng -> Tăng server
        if all(val > self.threshold for val in recent_forecast):
            self.current_servers += 1
            self.last_scale_time = current_time
            print(f"[{current_time}m] Scale-out: {self.current_servers} servers")
            
        # Logic Scale-in (tùy chọn): Nếu tải quá thấp (< 30% công suất dàn máy)
        elif all(val < (self.current_servers - 1) * self.server_capacity * 0.3 for val in recent_forecast):
            if self.current_servers > 1:
                self.current_servers -= 1
                self.last_scale_time = current_time
                print(f"[{current_time}m] Scale-in: {self.current_servers} servers")

        return self.current_servers