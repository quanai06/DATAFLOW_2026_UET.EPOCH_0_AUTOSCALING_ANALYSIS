import os
import sys
import subprocess
import time

def run_pipeline():
    # 1. Xác định các đường dẫn quan trọng
    # Lấy đường dẫn của file script hiện tại (scripts/pipeline_optimize.py)
    current_script_path = os.path.abspath(__file__)
    
    # Lấy thư mục gốc của dự án (project_root) bằng cách đi ngược lên 2 cấp
    project_root = os.path.dirname(os.path.dirname(current_script_path))
    
    # Đường dẫn đến thư mục chứa source code optimize
    src_optimize_path = os.path.join(project_root, 'src', 'optimize')
    
    # Các file cần chạy
    data_utils_script = os.path.join(src_optimize_path, 'data_utils.py')
    optimizer_script = os.path.join(src_optimize_path, 'runOptimizer.py')

    print(f"🚀 Bắt đầu Pipeline Tối ưu hóa Autoscaling...")
    print(f"📂 Working Directory: {project_root}")
    
    # QUAN TRỌNG: Thay đổi thư mục làm việc về project_root
    # Điều này giúp các lệnh trong code con (như pd.read_csv('results/...')) hoạt động đúng
    os.chdir(project_root)
    
    # Thêm src/optimize vào PYTHONPATH để runOptimizer có thể import universalOptimizer
    # (Mặc dù chạy subprocess file trực tiếp thường không cần, nhưng thêm vào cho an toàn)
    env = os.environ.copy()
    env["PYTHONPATH"] = src_optimize_path + os.pathsep + env.get("PYTHONPATH", "")

    # --- BƯỚC 1: CHUẨN BỊ DỮ LIỆU ---
    # Chạy data_utils.py để chuẩn bị dữ liệu
    print("🔄 Bước 1: Chuẩn bị dữ liệu..." )
    subprocess.run([sys.executable, data_utils_script], check=True, env=env)
    print("✅ Hoàn thành bước chuẩn bị dữ liệu.")
    print("Dữ liệu đã được lưu trong thư mục results/optimize/.")

    # --- BƯỚC 2: CHẠY TỐI ƯU HÓA ---
    # Chạy runOptimizer.py để thực hiện tối ưu hóa autoscaling
    print("🔄 Bước 2: Chạy tối ưu hóa autoscaling..." )
    subprocess.run([sys.executable, optimizer_script], check=True, env=env)
    print("✅ Hoàn thành bước tối ưu hóa autoscaling.")
    print("Các file kết quả được lưu trong thư mục results/optimize/.")

if __name__ == "__main__":
    run_pipeline()