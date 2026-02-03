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

Dự án thuộc khuôn khổ cuộc thi DATAFLOW 2026 - Câu lạc bộ Toán Tin HAMIC.

**Vấn đề cần giải quyết**:
Trong quản trị hệ thống đám mây, việc cấp phát tài nguyên tĩnh thường dẫn đến lãng phí khi thấp tải hoặc sập hệ thống (Overload) khi cao tải. 

**Ý tưởng và cách tiếp cận**:
Dự án này giải quyết hai bài toán cốt lõi:
1. **Time-Series Forecasting**: Dự báo lưu lượng truy cập (Request/Bytes) trong tương lai sử dụng các mô hình: XGBoost, LightGBM, LSTM.
2. **Cost Optimization**: Xây dựng thuật toán Autoscaling thông minh để cân bằng giữa chi phí thuê server và cam kết chất lượng dịch vụ (SLA).

**Giá trị thực tiễn**:
Giảm thiểu tối đa chi phí thuê máy chủ nhưng vẫn đảm bảo độ trễ thấp và tính sẵn sàng cao cho dịch vụ

## 📂 2. Dữ liệu:

**Nguồn**: Bộ dữ liệu nhật ký truy cập HTTP (Web Log) của máy chủ NASA trung tâm vũ trụ Kennedy (07/1995 - 08/1995)

**Mô tả trường dữ liệu chính** ```host, timestamp, request, status, bytes```

**Tiền xử lý đã thực hiện**


## 3. Mô hình và kiến trúc

**Kiến trúc tổng thể**

**Mô hình sử dụng**

**Chiến lược validation/training**

**Tránh data leakage bằng cách**

## ✅ 4. Đánh giá

**Metrics**

**Kết quả**

**Phân tích trade-off**

# 🚀 5. Triển khai và demo
## video cách chạy và giải thích chức năng của demo dự án  
- [Download from Google Drive](https://drive.google.com/file/d/FILE_ID/view?usp=sharing)

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
```
#chạy xử lí data
python src/scripts/pipeline_data.py
#load model
#chạy app rồi truy cập đường dẫn (http://localhost:8501)
python run_app.py 
```
### Nếu chỉ cần xem demo và sử dụng  
```
python run_app.py
```
## 6. Giới hạn và hướng phát triển

### Giới hạn hiện tại
- **Thiếu số liệu hạ tầng thực tế (CPU/RAM/Latency):** Dataset năm 1995 chỉ phản ánh log truy cập (requests/bytes) nên chưa thể kiểm chứng trực tiếp các chỉ số vận hành như **độ trễ (latency)**, **CPU/RAM usage**, **queue length**.
- **Môi trường mô phỏng đơn giản:** Chi phí và SLA penalty được mô hình hóa theo công thức tuyến tính; chưa mô phỏng đầy đủ các hiện tượng thực tế như cold start, network bottleneck, cache hit/miss.
- **Phụ thuộc vào chất lượng dự báo:** Predictive/Hybrid chịu ảnh hưởng bởi sai số mô hình (đặc biệt ở các đoạn spike), có thể dẫn đến over-provision hoặc under-provision nếu không có cơ chế bảo vệ.

### Hướng phát triển
- **Tích hợp Drift Detection:**  
  Nhận diện thay đổi hành vi theo thời gian (mùa vụ, giờ cao điểm, thay đổi hành vi người dùng) để:
  - cảnh báo khi phân phối traffic thay đổi,
  - tự động điều chỉnh tham số scaling hoặc trigger retrain model.
- **Triển khai thực tế lên Kubernetes:**  
  - Xuất các dự báo và khuyến nghị scaling ra **Custom Metrics API**,
  - Kết hợp với **HPA/VPA** hoặc custom controller để scale theo dự báo.
- **Bổ sung tín hiệu hệ thống thật:**  
  Thu thập CPU/RAM/Latency từ Prometheus/Grafana để:
  - đánh giá SLA theo latency thật,
  - tối ưu cost theo SLO/SLA thực tế thay vì chỉ requests/bytes.
- **Mô phỏng nâng cao:**  
  Thêm simulator cho queue/latency (Little’s Law hoặc mô hình hàng đợi) để phản ánh rõ hơn trade-off giữa **chi phí** và **trải nghiệm người dùng**.

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
