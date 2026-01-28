import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train_lgbm import GBDTTrainer
from src.train_xgboost import XGBoostTrainer

def train_lgbm():
    print("🚀 Bắt đầu Pipeline huấn luyện mô hình GBDT...")
    
    all_results = []
    timeframes = ['1m', '5m', '15m']
    targets = ['y_req', 'y_bytes_imp']

    for tf in timeframes:
        for target in targets:
            trainer = GBDTTrainer(timeframe=tf, target_col=target)
            
            result = trainer.train()
            
            result['Model_Type'] = 'LightGBM_GBDT' 
            all_results.append(result)
            
            print(f"✅ Hoàn thành: {result['Model_Type']} | {target} | {tf}")

    os.makedirs('results/lgbm', exist_ok=True)
    report_path = 'results/lgbm/lgbm_performance_report.csv'
    
    df_report = pd.DataFrame(all_results)
    
    cols = ['Model_Type', 'Target', 'Timeframe', 'RMSE', 'MSE', 'MAE', 'MAPE']
    df_report = df_report[cols]
    
    df_report.to_csv(report_path, index=False)
    
    print("\n" + "="*60)
    print(f"🏁 BÁO CÁO CHI TIẾT ĐÃ LƯU TẠI: {report_path}")
    print("="*60)

def train_xgboost():
    # Danh sách các khung thời gian và mục tiêu cần train theo đề bài
    timeframes = ['1m', '5m', '15m']
    targets = ['y_req_t1', 'y_bytes_imp_t1']
    
    all_results = []
    
    for tf in timeframes:
        for tg in targets:
            try:
                trainer = XGBoostTrainer(tf, tg)
                metrics = trainer.train()
                all_results.append(metrics)
                if tf == '5m':
                    trainer_q90 = XGBoostTrainer(tf, tg, objective='reg:quantileerror', quantile_alpha=0.9)
                    metrics_q90 = trainer_q90.train()
                    all_results.append(metrics_q90)
            except Exception as e:
                print(f"Lỗi khi train {tg} khung {tf}: {e}")
                
    # In bảng tổng hợp kết quả để đưa vào báo cáo
    if all_results:
        summary_df = pd.DataFrame(all_results)
        print("\n" + "="*50)
        print("BẢNG TỔNG HỢP KẾT QUẢ (BENCHMARKING)")
        print("="*50)
        print(summary_df)
        summary_df.to_csv('results/xgboost/xgboost_performance_report.csv', index=False)
        
if __name__ == "__main__":
    # train_lgbm()
    train_xgboost()