import math

class HybridOptimizer:  
    def __init__(self, cap_req, cap_bytes, k_base, alpha_5m, max_reduce,
                    panic_ratio, burst_add, patience, target_util, max_servers):
        self.cap_req = cap_req
        self.cap_bytes = cap_bytes
        self.eff_cap_req = cap_req * target_util
        self.eff_cap_bytes = cap_bytes * target_util
        self.k_base = k_base
        self.alpha_5m = alpha_5m
        self.max_reduce = max_reduce
        self.panic_ratio = panic_ratio
        self.burst_add = burst_add
        self.patience = patience
        self.target_util = target_util
        self.max_servers = max_servers
        self.current_servers = 1
        self.last_scale_time = -10
        self.low_load_counter = 0  # Đếm thời gian tải thấp liên tiếp

    # Tính số server cần thiết 
    def calculate_base_main(self, f15_req, f15_bytes, f5_req, f5_req_q90, f5_bytes, f5_bytes_q90):
        """Tầng 1 & 2: Bottleneck Principle áp dụng cho dự báo
        f15_req, f15_bytes: Dự báo 15 phút
        f5_req, f5_bytes: Dự báo 5 phút
        Returns: Số server cần thiết để đáp ứng cả 2 tầng
        Tính dựa trên hiệu suất mục tiêu thay vì sức tải tối đa của server
        Tầng 5 thay vì dùng buffer cố định, dùng công thức alpha
        """
        # N_base (15m)
        n_base = math.ceil(max(f15_req * self.k_base / self.eff_cap_req, 
                               f15_bytes * self.k_base / self.eff_cap_bytes))
        
        # N_main (5m)
        safe_req_5m = f5_req + self.alpha_5m * max(0, (f5_req_q90 - f5_req))
        safe_bytes_5m = f5_bytes + self.alpha_5m * max(0, (f5_bytes_q90 - f5_bytes))
        n_main = math.ceil(max(safe_req_5m / self.eff_cap_req, 
                               safe_bytes_5m / self.eff_cap_bytes))
        
        return max(n_base, n_main)

    def check_panic(self, act1_req, act1_bytes, f5_req_q90, f5_bytes_q90):
        """Tầng 3: Panic Trigger - Chống Spike
        Panic nếu 1 trong 2 loại tải vọt quá ngưỡng dự báo của tầng 5m
        Chống q90 = 0 bằng cách dùng max (q90, sức tải 1 server)
        """
        panic_req = act1_req > (max(f5_req_q90, self.eff_cap_req) * (1 + self.panic_ratio))
        panic_bytes = act1_bytes > (max(f5_bytes_q90, self.eff_cap_bytes) * (1 + self.panic_ratio))
        return panic_req or panic_bytes

    def step_hybrid (self, f15_req, f15_bytes, f5_req, f5_req_q90, f5_bytes, f5_bytes_q90, act1_req, act1_bytes, t):
        target_stable = self.calculate_base_main(f15_req, f15_bytes, f5_req, f5_req_q90, f5_bytes, f5_bytes_q90)
        is_panic = self.check_panic(act1_req, act1_bytes, f5_req_q90, f5_bytes_q90)
        # Nếu có panic, cộng thêm burst_add vào N_target
        if is_panic:
            # Tính số server cần thật sự để gánh Spike
            panic_needed_req = math.ceil(act1_req / self.cap_req)
            panic_needed_bytes = math.ceil(act1_bytes / self.cap_bytes)
            panic_needed = max(panic_needed_req, panic_needed_bytes)
            needed = max(target_stable , panic_needed + self.burst_add)

        else:
            needed = target_stable
        
        needed = min(needed, self.max_servers)  # Clamp to max_servers
        needed = max(needed, 1)  # Đảm bảo luôn có ít nhất 1 server

        # Logic Scale-out: Nhạy (2 phút)
        if needed > self.current_servers:
            # Nếu là panic: Luôn scale ngay lập tức khi số tải cần thiết thực sự vượt tất cả server hiện có
            if is_panic or (t - self.last_scale_time >= 2):
                self.current_servers = needed
                self.last_scale_time = t
                self.low_load_counter = 0

            else:
                # Nếu đang cooldown, giữ nguyên server nhưng cũng reset counter thấp tải
                self.low_load_counter = 0
        
        # Logic Scale-in: Thận trọng (Dùng counter để tạo Hysteresis)
        elif needed < self.current_servers:
            if t - self.last_scale_time >= 5:  # Chỉ bắt đầu đếm sau 5 phút từ lần scale cuối
                self.low_load_counter += 1
            if self.low_load_counter >= self.patience: 
                # Chỉ giảm nếu tải thấp duy trì đủ lâu
                # Chỉ giảm tối đa max_reduce máy mỗi lần để chống giật
                new_needed = max(needed, self.current_servers - self.max_reduce)
                self.current_servers = new_needed
                self.last_scale_time = t
                self.low_load_counter = 0
        
        else:
            # needed == current_servers -> Hệ thống ổn định
            self.low_load_counter = 0 # Reset counter

        return self.current_servers
 