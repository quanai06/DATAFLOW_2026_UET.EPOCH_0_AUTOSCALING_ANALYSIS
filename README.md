<div align="center">
    
  <img src="dataflow.png" width="200" alt="logo" />

  # Intelligent Autoscaling System - DataFlow 2026
  
  **Giải pháp Tối ưu hóa Tự động hóa việc cấp phát tài nguyên máy chủ (Autoscaling) dựa trên Học máy và Chiến lược lai (Hybrid Strategy).**
  
  ![Python](https://img.shields.io/badge/Python-3.12-red?logo=python&logoColor=white)
  ![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)
</div>

---

## 📖 Mục lục

- [Giới thiệu](#-giới-thiệu)
- [Mục tiêu và Đóng góp](#-mục-tiêu-và-đóng-góp)
- [Các Bước Hoạt Động](#-các-bước-hoạt-động)
- [Cài đặt](#-cài-đặt-và-chạy-dự-án)
- [Cách sử dụng](#-cách-sử-dụng)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Kết quả](#-kết-quả)


---

## 📖 Giới thiệu

Dự án thuộc khuôn khổ cuộc thi DATAFLOW 2026 - Câu lạc bộ Toán Tin HAMIC.
Trong quản trị hệ thống đám mây, việc cấp phát tài nguyên tĩnh thường dẫn đến lãng phí khi thấp tải hoặc sập hệ thống (Overload) khi cao tải. Dự án này giải quyết hai bài toán cốt lõi:
1. **Time-Series Forecasting**: Dự báo lưu lượng truy cập (Request/Bytes) trong tương lai sử dụng các mô hình: XGBoost, LightGBM, LSTM.
2. **Cost Optimization**: Xây dựng thuật toán Autoscaling thông minh để cân bằng giữa chi phí thuê server và cam kết chất lượng dịch vụ (SLA).

## 👥 Mục tiêu và Đóng góp

Dự án này được thực hiện bởi nhóm sinh viên từ trường [UET - VNU](https://uet.vnu.edu.vn) gồm 4 thành viên:

* **Lê Hoàng Quân** - Trưởng Nhóm
* **Vũ Hoàng Diệu Linh** 
* **Nguyễn Thị Hiền** 
* **Dương Trọng Nguyên** 

**Mục tiêu chính:**
Dựa trên việc phân tích nhật ký truy cập (log) của máy chủ WWW trong 2 tháng, từ đó xây dựng mô hình dự báo tải và thiết kế chính sách Autoscaling thông minh nhằm giảm thiểu lãng phí tài nguyên mà vẫn đảm bảo cam kết chất lượng dịch vụ (SLA).

## ✨ Các Bước Hoạt Động
Luồng xử lý của hệ thống (Pipeline) trải qua các bước sau:

1. **Ingest & Preprocessing**: Đọc log server, xử lý chuỗi thời gian, trích xuất đặc trưng (Requests/s, Bytes/s).
2. **Modeling**: Huấn luyện các mô hình học máy (LSTM, XGBoost, LightGBM) để dự báo tải.
Merging Data: Hợp nhất kết quả dự báo từ nhiều khung thời gian (1m, 5m, 15m) và dữ liệu thực tế.
3. **Simulation & Optimization (Core)**:
- Chạy mô phỏng Autoscaling trên dữ liệu lịch sử.
- Sử dụng Optuna để tối ưu hóa hàm mục tiêu (Cost Function).
4. **Evaluation**: Đánh giá hiệu quả trên tập dữ liệu kiểm thử (Test Set) dựa trên chi phí và tỷ lệ lỗi.
## 🚀 Cài đặt và Chạy dự án
**Yêu cầu hệ thống:**

Python 3.8 trở lên.

Các thư viện phụ thuộc trong file requirements.txt
### 1. Clone dự án
```bash
git clone 
cd 
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
Tạo file .env tại thư mục gốc để thiết lập chi phí giả định:
```bash
CAP_REQ=0.2
CAP_BYTES=5
COST_PER_SERVER_PER_MIN=100
COST_SOFT_SLA_PENALTY_REQ=200
COST_SOFT_SLA_PENALTY_BYTES=150
COST_HARD_SLA_PENALTY_REQ=5000
COST_HARD_SLA_PENALTY_BYTES=4000
COST_SCALING_EVENT=500
TARGET_UTIL=0.8
MAX_SERVERS=20
REQ_SPIKE_TH=1.4
BYTES_SPIKE_TH=30
SPIKE_WIN_K=5
SPIKE_WIN_N=3
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

## 📁 Cấu trúc dự án

```
2526-LTXLDL-Project-AIT2006-4-2.1/
├── README.md                    # Tài liệu dự án
├── requirements.txt             # Danh sách thư viện
├── figures/                     # Hình ảnh và biểu đồ
│   ├── Point_Maps_Keywords_HH_DD_MM_YYYY/
│   ├── Selected_Highlights_For_Report/
│   └── WordClouds_Keywords_HH_DD_MM_YYYY/
|   └── Early_Patients_Analysis.png
|   └── Advanced_lineplot_symptom_trend_daily_full_data.png
|   └── Early_Patients_Analysis.png
|   └── keyword_count_overtime.png
|   └── lineplot_symptom_trend_daily.png
|   └── lineplot_symptom_trend_hourly.png
|   └── spread_animation.gif
|   └── symptom_keyword_vs_weather_type.png
|   └── symptom_keyword_vs_wind_direction.png
|   └── symptom_keyword_vs_wind_speed.png
├── processed/                   # Dữ liệu đã xử lý
│   ├── keywords.csv
│   ├── Microblogs_Cleaned.csv
│   ├── Microblogs_With_Weather.csv
│   ├── population_cleaned.csv
│   ├── Weather_Cleaned.csv
│   ├── hourly_location_mappings/
│   └── stat_hourly/
├── raw/                         # Dữ liệu thô
│   ├── keywords.csv
│   ├── Microblogs.csv
│   ├── Population.csv
│   ├── symptom_keywords.txt
│   └── Weather.csv
│   └── Vastopolis_Map.png
├── reports/                     # Báo cáo và kết quả
│   └── qa_summary.csv
└── src/                         # Mã nguồn
    ├── Clean_data/              # Làm sạch dữ liệu
    │   └── clean_keywords.ipynb
    │   └── Clean_Microblogs.ipynb
    │   └── cleaned_population.ipynb
    │   └── cleaned_weather.ipynb
    │   └── generate_qa_reports.ipynb
    ├── Aggregate_data/          # Tổng hợp dữ liệu
    │   └── location.ipynb
    │   └── merge_microblogs&weather.ipynb
    │   └── stat_hourly.ipynb
    ├── Visualization/           # Trực quan hóa
    │   └── bar_chart_2122h.ipynb
    │   └── correlation_blog_weather.ipynb
    │   └── Feature.ipynb
    │   └── line_graph.ipynb
    │   └── Point_Maps.ipynb
    │   └── Spread_Animation.ipynb
    │   └── Visualization_hours.ipynb
    │   └── Wordcloud.ipynb
    └── Bonus/                 # Tính năng bổ sung
        └── advanced_full_data.ipynb
        └── find_origin.ipynb 
        └── Speed_Animation.ipynb
    
```

## 📊 Kết quả

Dự án cung cấp các kết quả sau:
- **Thống kê theo giờ**: Phân tích vị trí, số lượng(symstom keyword theo từng yếu tố),số lượng người bệnh theo thời gian
- **Chọn ra thời điểm đặc biệt**: Thời gian dịch nhỏ nhất, đỉnh dịch, sau đỉnh dịch
- **Word Cloud**: Trực quan hóa từ khóa phổ biến
- **Bản đồ tâm dịch**: Xác định vị trí bùng phát dịch bệnh
- **Báo cáo tổng hợp**: Tóm tắt phân tích trong `reports/qa_summary.csv`

---

<div align="center">
  <p>Được phát triển bởi nhóm UET_EPOCH0</p>
  <p>Trường Đại học Công nghệ - Đại học Quốc gia Hà Nội</p>
</div>
