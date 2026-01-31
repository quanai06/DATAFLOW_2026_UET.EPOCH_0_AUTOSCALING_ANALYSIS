import math

# Universal Optimizer kết hợp cả 3 chế độ: Reactive, Predictive, Hybrid
# Sử dụng HybridOptimizer đã có cho chế độ hybrid
# Tính số server cần thiết dựa trên chế độ đã chọn
class UniversalOptimizer:
    def __init__(self, mode, cap_req, cap_bytes, target_util, max_servers, **params):
        self.mode = mode
        self.cap_req = cap_req
        self.cap_bytes = cap_bytes
        self.eff_cap_req = cap_req * target_util
        self.eff_cap_bytes = cap_bytes * target_util
        self.target_util = target_util
        self.max_servers = max_servers
        self.params = params
        self.current_servers = 1
        self.last_scale_time = -10
        self.out_counter = 0 #đếm để tăng máy
        self.in_counter = 0 #đếm để giảm máy

    def calculate_needed(self, row, t):
        
        if self.mode == "reactive":
            # Chỉ dùng thực tế (Actual)
            n_req = math.ceil(row.act_1m_req / self.eff_cap_req)
            n_bytes = math.ceil(row.act_1m_bytes / self.eff_cap_bytes)
            return max(n_req, n_bytes), False  # Không có is_panic trong chế độ reactive
        
        elif self.mode == "predictive":
            # Chỉ dùng dự báo (Tầng 1m dùng Q90)
            n_15m = math.ceil(max(row.predicted_15m_req * self.params['k_base'] / self.eff_cap_req, 
                                  row.predicted_15m_bytes * self.params['k_base'] / self.eff_cap_bytes))
            n_5m = math.ceil(max(row.predicted_5m_req * (1 + self.params['alpha_5m']) / self.eff_cap_req,
                                 row.predicted_5m_bytes * (1 + self.params['alpha_5m']) / self.eff_cap_bytes))
            n_1m = math.ceil(max(row.predicted_1m_q90_req / self.eff_cap_req,
                                 row.predicted_1m_q90_bytes / self.eff_cap_bytes)) 
            return max(n_15m, n_5m, n_1m), False  # Không có is_panic trong chế độ predictive
        
        else:  # hybrid
            # N_base (15m)
            n_base = math.ceil(max(row.predicted_15m_req * self.params['k_base'] / self.eff_cap_req, 
                                row.predicted_15m_bytes * self.params['k_base'] / self.eff_cap_bytes))
            
            # N_main (5m)
            safe_req_5m = row.predicted_5m_req + self.params['alpha_5m'] * max(0, (row.predicted_5m_q90_req - row.predicted_5m_req))
            safe_bytes_5m = row.predicted_5m_bytes + self.params['alpha_5m'] * max(0, (row.predicted_5m_q90_bytes - row.predicted_5m_bytes))
            n_main = math.ceil(max(safe_req_5m / self.eff_cap_req, 
                                safe_bytes_5m / self.eff_cap_bytes))
            
            panic_req = row.act_1m_req > (max(row.predicted_5m_q90_req, self.eff_cap_req) * (1 + self.params['panic_ratio']))
            panic_bytes = row.act_1m_bytes > (max(row.predicted_5m_q90_bytes, self.eff_cap_bytes) * (1 + self.params['panic_ratio']))
            is_panic = panic_req or panic_bytes
            
            stable_needed = max(n_base, n_main)

            if is_panic:
                panic_needed_req = math.ceil(row.act_1m_req / self.eff_cap_req)
                panic_needed_bytes = math.ceil(row.act_1m_bytes / self.eff_cap_bytes)
                panic_needed = max(panic_needed_req, panic_needed_bytes)
                needed = max(stable_needed, panic_needed + self.params.get("burst_add", 0))
            else:
                needed = stable_needed

            return needed, is_panic


    def step(self, row, t):
        # 1. Tính toán nhu cầu (Mỗi mode sẽ có cách tính needed và is_panic riêng)
        # Lưu ý: Sửa calculate_needed để luôn trả về (needed, is_panic)
        needed, is_panic = self.calculate_needed(row, t)
        
        # Lấy các tham số cấu hình tùy theo mode (nếu không có thì lấy mặc định)
        if self.mode == "reactive":
            out_patience = self.params.get("out_patience", 3)
            in_patience  = self.params.get("in_patience", 5)
            max_reduce   = self.params.get("max_reduce", 1)
        elif self.mode == "predictive":
            out_patience = 0
            in_patience  = self.params.get("in_patience", 5)
            max_reduce   = self.params.get("max_reduce", 1)
        else:  # hybrid
            out_patience = self.params.get("out_patience", 2)   # non-panic
            in_patience  = self.params.get("in_patience", 5)
            max_reduce   = self.params.get("max_reduce", 1)

        # SCALE-OUT
        if needed > self.current_servers:
            self.in_counter = 0

            # HYBRID panic hoặc PREDICTIVE: tăng ngay
            if (self.mode == "predictive") or (self.mode == "hybrid" and is_panic):
                self.current_servers = needed
                self.last_scale_time = t
                self.out_counter = 0
            else:
                # reactive / hybrid non-panic: cần out_patience
                self.out_counter += 1
                if self.out_counter >= out_patience:
                    self.current_servers = needed
                    self.last_scale_time = t
                    self.out_counter = 0


        # TRƯỜNG HỢP 2: GIẢM MÁY (SCALE-IN)
        elif needed < self.current_servers:
            self.out_counter = 0

            if t - self.last_scale_time < 5:
                self.in_counter = 0
                return self.current_servers

            self.in_counter += 1
            if self.in_counter >= in_patience:
                # Giảm từ từ để chống giật máy (Conservative Step-down)
                new_val = max(needed, self.current_servers - max_reduce)
                self.current_servers = new_val
                self.last_scale_time = t
                self.in_counter = 0
        
        # TRƯỜNG HỢP 3: ỔN ĐỊNH
        else:
            self.out_counter = 0
            self.in_counter = 0

        # Ràng buộc vật lý (Clamping)
        self.current_servers = max(1, min(self.current_servers, self.max_servers))
        
        return self.current_servers