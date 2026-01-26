import pandas as pd
import numpy as np
import math
import os
import optuna

class HybridOptimizer:
    def __init__(self, cap_req, cap_bytes, k_base, buffer_5m, panic_threshold, burst_add, patience):
        self.cap_req = cap_req
        self.cap_bytes = cap_bytes
        # Các tham số sẽ được Optuna tối ưu
        self.k_base = k_base
        self.buffer_5m = buffer_5m
        self.panic_threshold = panic_threshold
        self.burst_add = burst_add 
        self.patience = patience # Số chu kỳ chờ trước khi scale-in
        
        self.current_servers = 1
        self.last_scale_time = -10
        self.low_load_counter = 0 # Đếm để xử lý hysteresis

    def calculate_base_main(self, f15_req, f15_bytes, f5_req, f5_bytes):
        """Tầng 1 & 2: Bottleneck Principle áp dụng cho dự báo"""
        # N_base (15m)
        n_base = math.ceil(max(f15_req * self.k_base / self.cap_req, 
                               f15_bytes * self.k_base / self.cap_bytes))
        # N_main (5m)
        n_main = math.ceil(max(f5_req * (1 + self.buffer_5m) / self.cap_req, 
                               f5_bytes * (1 + self.buffer_5m) / self.cap_bytes))
        return max(n_base, n_main)

    def check_panic(self, act1_req, f5_req):
        """Tầng 3: Panic Trigger - Chống Spike"""
        if act1_req > (f5_req + self.panic_threshold):
            return True
        return False

    def step(self, f15_req, f15_bytes, f5_req, f5_bytes, act1_req, t):
        target_stable = self.calculate_base_main(f15_req, f15_bytes, f5_req, f5_bytes)
        
        # Nếu có panic, cộng thêm burst_add vào N_target
        if self.check_panic(act1_req, f5_req):
            needed = target_stable + self.burst_add
        else:
            needed = target_stable

        # Logic Scale-out: Nhạy (2 phút)
        if needed > self.current_servers:
            if t - self.last_scale_time >= 2:
                self.current_servers = needed
                self.last_scale_time = t
                self.low_load_counter = 0
        
        # Logic Scale-in: Thận trọng (Dùng counter để tạo Hysteresis)
        elif needed < self.current_servers:
            self.low_load_counter += 1
            if self.low_load_counter >= self.patience: # Chỉ giảm nếu tải thấp duy trì đủ lâu
                self.current_servers = needed
                self.last_scale_time = t
                self.low_load_counter = 0
                
        return self.current_servers
 