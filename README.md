<div align="center">
    
  <img src="dataflow.png" width="200" alt="logo" />

  # Chủ đề: Autoscaling Analysis - DataFlow 2026
  
  **Giải pháp Tối ưu hóa Tự động hóa việc cấp phát tài nguyên máy chủ (Autoscaling) dựa trên Học máy và Chiến lược lai (Hybrid Strategy).**
  
  ![Python](https://img.shields.io/badge/Python-3.12-red?logo=python&logoColor=white)
  ![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)
</div>

---

## 📖 Mục lục

- [1. Tóm tắt](#-1-tóm-tắt)
- [2. Dữ liệu](#-2-dữ-liệu)
- [3. Mô hình và kiến trúc](#-3-mô-hình-và-kiến-trúc)
- [4. Đánh giá](#-4-đánh-giá)
- [5. Triển khai và Demo](#-5-triển-khai-và-demo)
- [6. Giới hạn và Hướng phát triển](#-6-giới-hạn-và-hướng-phát-triển)
- [7. Tác động và ứng dụng](#-7-tác-động-và-ứng-dụng)
- [8. Tác giả và Giấy phép](#-8-tác-giả-và-giấy-phép)


---

## 📖 1. Tóm tắt

Dự án thuộc khuôn khổ cuộc thi **DATAFLOW 2026** — Câu lạc bộ Toán Tin **HAMIC**.

### Bối cảnh & vấn đề
Trong quản trị hệ thống đám mây, **cấp phát tài nguyên tĩnh** thường gây:
- **Lãng phí** khi thấp tải (over-provision)
- **Sập hệ thống / timeout** khi cao tải (overload)

### Ý tưởng & cách tiếp cận
Dự án giải quyết 2 bài toán chính:

1. **Time-Series Forecasting**  
   Dự báo lưu lượng truy cập trong tương lai theo 2 tín hiệu:
   - **Requests** (số request theo thời gian)
   - **Bytes** (tổng dữ liệu truyền tải theo thời gian)  
   Mô hình sử dụng: **XGBoost, LightGBM, LSTM**.

2. **Cost Optimization (Autoscaling Policy)**  
   Xây dựng thuật toán autoscaling nhằm tối ưu:
   - **Chi phí thuê server**
   - **Mức độ vi phạm SLA** (soft/hard overload)
   - **Tính ổn định vận hành** (hạn chế flapping qua scaling event cost)

### Giá trị thực tiễn
Giảm chi phí thuê máy chủ nhưng vẫn đảm bảo **độ trễ thấp** và **tính sẵn sàng cao** cho dịch vụ.

---

## 📂 2. Dữ liệu

### Nguồn dữ liệu
Bộ **HTTP Web Log** của máy chủ NASA (Kennedy Space Center), giai đoạn **07/1995 – 08/1995**.

### Trường dữ liệu chính
`host, timestamp, request, status, bytes`

### Tiền xử lý đã thực hiện
- Làm sạch log (chuẩn hóa format request, status, bytes; loại bỏ dòng lỗi)
- Resample theo nhiều tầng thời gian: **1m / 5m / 15m**
- Feature engineering cho time-series:
  - time features: `hour, weekday, is_weekend, ...`
  - traffic features: rolling/lag (`req_lag`, `bytes_lag`, rolling mean/std, ...)
  - quality signals: `error_rate`, `server_error_rate`, `bytes_missing_rate`, ...
- Tách dữ liệu train/valid/test theo **time-based split** để tránh leakage.


---

## 🏗️ 3. Mô hình và kiến trúc

### Kiến trúc tổng thể
Pipeline gồm 2 khối độc lập:

1. **Forecasting Service**  
   Input: dữ liệu lịch sử (1m/5m/15m) + feature  
   Output: dự báo `pred_req_t1`, `pred_bytes_imp_t1` cho các horizon tương ứng

2. **Autoscaling Optimizer**  
   Input: `act_1m_req/bytes` + `pred_5m` + `pred_15m` (tùy chiến lược)  
   Output: `recommended_servers` theo từng phút, kèm scaling events và SLA penalties

### Mô hình sử dụng
- **XGBoost**: baseline mạnh, tốc độ train nhanh, ổn định
- **LightGBM**: nhanh, phù hợp feature tabular/rolling/lag
- **LSTM**: nắm trend theo chuỗi, phù hợp khi pattern theo thời gian rõ

### Chiến lược validation/training
- **Time-based split** (không shuffle)
- Đánh giá theo từng horizon (1m/5m/15m)
- So sánh mô hình theo:
  - Forecast accuracy (MAPE/RMSE/MAE/MSE)
  - Downstream autoscaling cost (chi phí + SLA)

### Tránh data leakage
- Không dùng thông tin tương lai (t+1 trở đi) trong feature tại thời điểm t
- Lag/rolling được tính thuần từ lịch sử
- Train/valid/test được tách theo mốc thời gian liên tục

---

## ✅ 4. Đánh giá

### Metrics
**Forecasting**
- MAE / RMSE / MSE / MAPE 

**Autoscaling**
- **Total Cost** = server cost + SLA penalty (soft/hard) + scaling events cost
- **Soft Overload**: phần vượt quá “ngưỡng an toàn” (effective capacity)
- **Hard Overload**: phần vượt quá “ngưỡng cứng” (hard capacity)
- **Scaling Events**: số lần thay đổi số server (đánh đổi ổn định)

### Kết quả chính (tóm tắt)
- **Hiệu quả chi phí (Cost Efficiency):**  
  Chiến lược **Hybrid** đạt hiệu quả chi phí tốt nhất.  
  Cụ thể, **Hybrid (LSTM)** giảm khoảng **30.3% tổng chi phí** so với **Reactive**
  (**~11.74M vs ~16.84M**).  
  So với **Predictive**, Hybrid vẫn tiết kiệm thêm khoảng **12–17%**, cho thấy việc kết hợp
  giữa dự báo và cơ chế phản ứng theo thời gian thực giúp tối ưu phân bổ tài nguyên.

- **Độ an toàn & độ tin cậy SLA (Reliability):**  
  **Reactive** gây ra **Hard Overload ~472–479 phút**.  
  **Predictive** giảm xuống còn **~101–125 phút**.  
  **Hybrid** tiếp tục cải thiện rõ rệt, đặc biệt **Hybrid (LightGBM)** chỉ còn **~83 phút**  
  → giảm khoảng **82–83%** vi phạm nghiêm trọng so với Reactive.

  > Lưu ý: Soft/Hard overload được tính theo **tổng mức chênh lệch tải vs năng lực** (diện tích vượt tải),
  > không phải số lần vi phạm. Vì vậy Hybrid có thể xuất hiện nhiều thời điểm quá tải nhỏ lẻ,
  > nhưng tổng mức vượt tải vẫn thấp hơn → kiểm soát rủi ro tốt hơn.

- **Hành vi vận hành (System Behavior):**  
  **Hybrid** có số lần **Scaling Events** cao nhất (**~900–1,100**) so với:
  - Reactive (**~265**)
  - Predictive (**~330–580**)  
  Điều này cho thấy Hybrid điều chỉnh thường xuyên hơn để bám sát biến động tải.
  Đổi lại, hệ thống duy trì mức vượt tải nhỏ và ổn định hơn, nhưng tăng tần suất thay đổi tài nguyên.

### Phân tích trade-off
- **Reactive:** ổn định (ít scaling events) nhưng dễ “đuối” khi spike → hard overload cao
- **Predictive:** giảm overload mạnh, nhưng phụ thuộc sai số dự báo
- **Hybrid:** cân bằng tốt nhất giữa **cost** và **reliability**, đổi lại **scaling events** tăng


## 🚀 5. Triển khai và demo
### video cách chạy và giải thích chức năng của demo dự án  
- [video cách chạy và sử dụng](https://drive.google.com/file/d/FILE_ID/view?usp=sharing)

**Yêu cầu hệ thống:**

Python 3.10 trở lên, tối thiểu 8gb RAM 

Các thư viện phụ thuộc trong file requirements.txt

### 1. Clone dự án
```bash
git clone https://github.com/quanai06/DATAFLOW_2026_UET.EPOCH_0_AUTOSCALING_ANALYSIS.git
cd DATAFLOW_2026_UET.EPOCH_0_AUTOSCALING_ANALYSIS
```

### 2. Tạo môi trường ảo (khuyến nghị)
```bash
python -m venv .venv
# Trên Linux/Macos
source .venv/bin/activate  
# Trên Windows: 
source .venv\Scripts\activate
```

### 3. Cài đặt thư viện
```bash
pip install -r requirements.txt
```
### 4. Cấu hình môi trường
Tạo file .env bằng cách sao chép file mẫu .env.example tại thư mục gốc để thiết lập chi phí giả định:
```bash
cp .env.example .env
```
## 📋 Cách sử dụng

### Chạy toàn bộ pipeline để kiểm chứng
Chạy lần lượt các file theo thứ tự sau:
Chú ý: Nếu huấn luyện lại cả 3 model, thời gian xấp xỉ 2 tiếng.
```bash
# chạy xử lí data
python src/scripts/pipeline_data.py
# scripts train model (local)
python scripts/pipeline_training.py
# scripts test (load model)
python scripts/pipeline_testing.py
# scripts optimize
python scripts/pipeline_optimize.py
# chạy app rồi truy cập đường dẫn (http://localhost:8501)
python run_app.py 
```
### Nếu chỉ cần xem demo và sử dụng  
```bash 
python run_app.py
```
## 6. Giới hạn và hướng phát triển

### Giới hạn hiện tại
- **Thiếu số liệu hạ tầng thực tế (CPU/RAM/Latency):** Dataset năm 1995 chỉ phản ánh log truy cập (requests/bytes) nên chưa thể kiểm chứng trực tiếp các chỉ số vận hành như **độ trễ (latency)**, **CPU/RAM usage**, **queue length**.
- **Môi trường mô phỏng đơn giản:** Chi phí và SLA penalty được mô hình hóa theo công thức tuyến tính; chưa mô phỏng đầy đủ các hiện tượng thực tế như cold start, network bottleneck, cache hit/miss.
- **Phụ thuộc vào chất lượng dự báo:** Predictive/Hybrid chịu ảnh hưởng bởi sai số mô hình (đặc biệt ở các đoạn spike), có thể dẫn đến over-provision hoặc under-provision nếu không có cơ chế bảo vệ.

### Hướng phát triển

Để nâng cao tính hoàn thiện và giá trị ứng dụng, nhóm đề xuất các hướng phát triển theo **ba trục chính**:
**(1) độ tin cậy thuật toán**, **(2) chất lượng đánh giá**, và **(3) khả năng triển khai**.

### 1. Nâng cao độ tin cậy của phát hiện nhiễu và bất thường (Spike/Anomaly)
Hiện tại, dự án phát hiện spike theo hướng **rule-based** (ngưỡng cố định) và/hoặc **ngưỡng động**.  
Trong giai đoạn tiếp theo, có thể chuẩn hoá thành mô-đun độc lập **`SpikeDetector`** (tách khỏi logic autoscaling), cho phép cấu hình nhiều phương án:

- **Ngưỡng cố định theo request/bytes** (dễ giải thích, phù hợp demo).
- **Ngưỡng động theo forecast quantile** (phù hợp predictive/hybrid).
- **Statistical detector** (z-score / robust z-score) hoặc **change-point detection** (khi cần nâng mức “anomaly”).

### 2. Chuẩn hoá mục tiêu tối ưu và thước đo theo chuẩn vận hành (Operational Metrics)
Chi phí hiện được tổng hợp từ **server-minutes**, **SLA penalty** (soft/hard overload) và **số lần scaling**.  
Hệ thống có thể phát triển theo hướng các chỉ số vận hành rõ ràng hơn:

- **Tách KPI theo SLO/SLA**: ví dụ `hard overload minutes ≤ X`, `soft overload minutes ≤ Y`, hoặc quy đổi sang **độ trễ giả lập (simulated latency)**.
- **Constrained optimization**: giữ mức vi phạm SLA trong giới hạn, sau đó **tối ưu chi phí** trong ràng buộc đó.
- **Multi-objective (Pareto frontier)**: đồng thời tối ưu **chi phí**, **SLA**, và **scaling events** để cung cấp góc nhìn trade-off minh bạch.

### 3. Củng cố khả năng triển khai và mở rộng kiểm thử (Robustness & Testing)
Để hệ thống mạnh mẽ hơn và sẵn sàng triển khai thực tế, cần bổ sung:

- **Logging/Trace**: ghi nhận thời gian chạy mô phỏng, kiểm soát lỗi đọc dữ liệu, và thông báo lỗi thân thiện.
- **Đánh giá trên đa dạng mẫu traffic**: kiểm thử với burst, seasonality mạnh, hoặc regime shift.
- **Synthetic data generation**: tạo dữ liệu giả lập để kiểm thử edge-case như:
  - missing minutes kéo dài,
  - spike liên tục,
  - thay đổi phân phối đột ngột (distribution shift).

---

## 7. Tác động và ứng dụng

### Lợi ích
- **Giảm chi phí vận hành:** Tự động cân bằng giữa số lượng server và mức tải thực tế, hạn chế over-provision kéo dài.
- **Giảm rủi ro vi phạm SLA:** Hybrid strategy có cơ chế “panic/burst” để phản ứng nhanh khi xuất hiện spike, giảm hard overload.
- **Chống flapping tốt hơn:** Có cooldown/patience và penalty cho scaling events giúp hệ thống ổn định, tránh scale lên/xuống liên tục.
- **Dễ mở rộng & tích hợp:** Thiết kế tách rời pipeline (forecast → recommend) giúp thay thế model (LGBM/XGBoost/LSTM) mà không đổi logic scaling.

### Kịch bản triển khai (Deployment Scenarios)
1. **Website/Portal có traffic theo giờ (giờ hành chính, cuối tuần)**
   - Dùng predictive scaling (5m/15m) để scale trước giờ cao điểm.
   - Reactive xử lý các dao động nhỏ và sai số dự báo.

2. **Hệ thống thương mại điện tử (flash sale / campaign)**
   - Dùng hybrid: dự báo để chuẩn bị baseline capacity.
   - Khi spike bất ngờ, kích hoạt panic_ratio + burst_add để scale-out nhanh, giảm downtime/timeout.

3. **API service/B2B theo hợp đồng SLA**
   - Ưu tiên hạn chế hard overload bằng cách tăng weight penalty.
   - Theo dõi drift để phát hiện thay đổi pattern (khách mới, tích hợp mới), tránh “lệch mùa vụ”.

4. **Triển khai trên Kubernetes**
   - Forecast service xuất metric dự báo (req/bytes) qua Custom Metrics.
   - Autoscaler controller đọc metric và đề xuất replicas cho HPA.
   - Kết hợp Prometheus để giám sát latency và điều chỉnh penalty theo SLO.

5. **Nền tảng nội bộ (microservices)**
   - Mỗi service có optimizer riêng (CAP_REQ/CAP_BYTES khác nhau).
   - Có thể chạy A/B giữa reactive vs hybrid để so sánh cost và SLA theo thời gian.



## 👥 8. Tác giả và giấy phép

Dự án này được thực hiện bởi nhóm sinh viên từ trường [UET - VNU](https://uet.vnu.edu.vn) gồm 4 thành viên:

* **Lê Hoàng Quân** - Trưởng Nhóm
* **Vũ Hoàng Diệu Linh** 
* **Nguyễn Thị Hiền** 
* **Dương Trọng Nguyên** 

<div align="center">
  <p>Được phát triển bởi nhóm UET_EPOCH0</p>
  <p>Trường Đại học Công nghệ - Đại học Quốc gia Hà Nội</p>
</div>
