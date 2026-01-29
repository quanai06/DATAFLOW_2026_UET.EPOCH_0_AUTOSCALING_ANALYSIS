import math
from hybridopt import HybridOptimizer

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

        if mode == "hybrid":
            self.hybrid_opt = HybridOptimizer(
                cap_req=cap_req,
                cap_bytes=cap_bytes,
                k_base=params['k_base'],
                alpha_5m=params['alpha_5m'],
                max_reduce=params['max_reduce'],
                panic_ratio=params['panic_ratio'],
                burst_add=params['burst_add'],
                patience=params['in_patience'],
                target_util=target_util,
                max_servers=max_servers
            )

    def calculate_needed(self, row, t):
        
        if self.mode == "reactive":
            # Chỉ dùng thực tế (Actual)
            n_req = math.ceil(row.act_1m_req / self.eff_cap_req)
            n_bytes = math.ceil(row.act_1m_bytes / self.eff_cap_bytes)
            return max(n_req, n_bytes)
        
        elif self.mode == "predictive":
            # Chỉ dùng dự báo (Tầng 1m dùng Q90)
            n_15m = math.ceil(max(row.predicted_15m_req * self.params['k_base'] / self.eff_cap_req, 
                                  row.predicted_15m_bytes * self.params['k_base'] / self.eff_cap_bytes))
            n_5m = math.ceil(max(row.predicted_5m_req * (1 + self.params['alpha_5m']) / self.eff_cap_req,
                                 row.predicted_5m_bytes * (1 + self.params['alpha_5m']) / self.eff_cap_bytes))
            n_1m = math.ceil(max(row.predicted_1m_q90_req / self.eff_cap_req,
                                 row.predicted_1m_q90_bytes / self.eff_cap_bytes)) 
            return max(n_15m, n_5m, n_1m)
        
        else:  # hybrid
            return self.hybrid_opt.step_hybrid(
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

    def step(self, row, t):
        needed = self.calculate_needed(row, t)
        
        # --- LOGIC ĐIỀU KHIỂN RIÊNG CHO TỪNG CHẾ ĐỘ ---
        
        if self.mode == 'reactive':
            # Tăng máy: Cần quá tải liên tục X phút (ví dụ 3 phút) mới tăng
            if needed > self.current_servers:
                self.out_counter += 1
                if self.out_counter >= self.params['out_patience']:
                    self.current_servers = needed
                    self.out_counter = 0
                    self.in_counter = 0
                    self.last_scale_time = t

            # Giảm máy: Cần thấp liên tục Y phút mới giảm
            elif needed < self.current_servers:
                self.in_counter += 1
                if self.in_counter >= self.params['in_patience']:
                    max_reduce = self.params.get('max_reduce', 100) 
                    self.current_servers = max(needed, self.current_servers - max_reduce)
                    self.in_counter = 0
                    self.out_counter = 0
                    self.last_scale_time = t
            else:
                self.out_counter = 0
                self.in_counter = 0

        elif self.mode == 'predictive':
            # Tăng máy: THẤY DỰ BÁO TĂNG LÀ TĂNG NGAY (Không cần out_counter)
            if needed > self.current_servers:
                self.current_servers = needed
                self.in_counter = 0
                self.last_scale_time = t
            # Giảm máy: Vẫn cần thận trọng 
            elif needed < self.current_servers:
                self.in_counter += 1
                if self.in_counter >= self.params['in_patience']:
                    self.current_servers = needed
                    self.in_counter = 0
                    self.last_scale_time = t
        # (Chế độ Hybrid giữ nguyên cơ chế: Tăng ngay khi Panic, Giảm chậm)
        else:  # hybrid
            self.current_servers = self.calculate_needed(row, t)
            return self.current_servers

        self.current_servers = min(self.current_servers, self.max_servers)
        self.current_servers = max(self.current_servers, 1)

        return self.current_servers