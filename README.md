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

## 2. Dữ liệu:

**Nguồn**: Bộ dữ liệu nhật ký truy cập HTTP (Web Log) của máy chủ NASA trung tâm vũ trụ Kennedy (07/1995 - 08/1995)

**Mô tả trường dữ liệu chính** ```host, timestamp, request, status, bytes```

**Tiền xử lý đã thực hiện**


## 3. Mô hình và kiến trúc

**Kiến trúc tổng thể**

**Mô hình sử dụng**

**Chiến lược validation/training**

**Tránh data leakage bằng cách**

## 4. Đánh giá

**Metrics**

**Kết quả**

**Phân tích trade-off**

# 5. Triển khai và demo
**Yêu cầu hệ thống:**

Python 3.8 trở lên.

Các thư viện phụ thuộc trong file requirements.txt

### 1. Clone dự án
```bash
git clone https://github.com/quanai06/DATAFLOW_2026_UET.EPOCH_0_AUTOSCALING_ANALYSIS.git
cd DATAFLOW_2026_UET.EPOCH_0_AUTOSCALING_ANALYSIS
```

### 2. Tạo môi trường ảo (khuyến nghị)
```bash
python -m venv venv
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

### Chạy toàn bộ pipeline
```
Chạy lần lượt các file theo thứ tự sau: 

```

### Xử lý dữ liệu
1. **Làm sạch dữ liệu**: Chạy `src/Clean_data/.......`
2. **Tổng hợp dữ liệu**: Chạy `src/Aggregate_data/.......`
3. **Trực quan hóa**: Chạy `src/Visualization/.....`

### Tạo báo cáo
```bash
python src/Clean_data/generate_qa_reports.ipynb
```
## 6. Giới hạn và hướng phát triển

* **Giới hạn hiện tại:** Dữ liệu năm 1995 chưa có các thông số về CPU/RAM thực tế để kiểm chứng độ trễ hệ thống (Latency).

* **Kế hoạch cải tiến:** Tích hợp Drift Detection để nhận diện sự thay đổi hành vi người dùng theo mùa vụ, triển khai trực tiếp lên Kubernetes thông qua Custom Metrics API.

## 7. Tác động và ứng dụng

**Lợi ích**

**Kịch bản triển khai**


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
