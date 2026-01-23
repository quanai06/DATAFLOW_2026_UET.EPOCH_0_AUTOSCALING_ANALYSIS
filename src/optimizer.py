import math

class AutoscalingOptimizer:
    def __init__(self, capacity_req, capacity_bytes, cooldown_minutes=5, target_util=0.7):
        self.capacity_req = capacity_req # Sức tải mỗi server (req/phút)
        self.capacity_bytes = capacity_bytes # Sức tải mỗi server (bytes/phút)

        # Thay vì dùng 1 ngưỡng threshold cố định, ta dùng một hệ số để luôn có dư một khoảng cho các biến động bất ngờ
        self.target_util = target_util 
        self.cooldown_period = cooldown_minutes # Cooldown tránh flapping (phút)

        self.current_servers = 1
        self.last_scale_time = -cooldown_minutes # Đảm bảo có thể scale ngay lúc đầu
        self.decision_history = []
        self.min_servers = 1

    def calculate_needed_servers(self, forecast_req, forecast_bytes):
        # Tính số server cần thiết dựa trên dự báo và công suất
        n_req = math.ceil(forecast_req / (self.capacity_req * self.target_util))
        n_bytes = math.ceil(forecast_bytes / (self.capacity_bytes * self.target_util))

        return max(n_req, n_bytes, self.min_servers)

    def check_scaling(self, series_req, series_bytes, current_time):
        # Đảm bảo có đủ dữ liệu để tính trung bình 5 phút khi scale-in
        if len(series_req) < 1: return self.current_servers
        
        needed_now = self.calculate_needed_servers(series_req[-1], series_bytes[-1])

        # SCALE-OUT: Tăng máy chủ (Ưu tiên hiệu năng, nhạy bén)
        if needed_now > self.current_servers:
            if current_time - self.last_scale_time >= 2: 
                self.current_servers = needed_now
                self.last_scale_time = current_time
                print(f"[{current_time}m] >> SCALE-OUT to {self.current_servers} servers")
        
        # SCALE-IN: Giảm máy chủ (Thận trọng, tránh dao động)
        elif needed_now < self.current_servers:
            if current_time - self.last_scale_time >= self.cooldown_period:
                if len(series_req) >= 5:
                    avg_req = sum(series_req[-5:]) / 5
                    avg_bytes = sum(series_bytes[-5:]) / 5
                    stable_need = self.calculate_needed_servers(avg_req, avg_bytes)
                    if stable_need < self.current_servers:
                        self.current_servers = stable_need
                        self.last_scale_time = current_time
                        print(f"[{current_time}m] << SCALE-IN to {self.current_servers} servers")
        
        return self.current_servers