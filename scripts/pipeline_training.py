import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train_lgbm import GBDTTrainer

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

    os.makedirs('results', exist_ok=True)
    report_path = 'results/lgbm_performance_report.csv'
    
    df_report = pd.DataFrame(all_results)
    
    cols = ['Model_Type', 'Target', 'Timeframe', 'RMSE', 'MSE', 'MAE', 'MAPE']
    df_report = df_report[cols]
    
    df_report.to_csv(report_path, index=False)
    
    print("\n" + "="*60)
    print(f"🏁 BÁO CÁO CHI TIẾT ĐÃ LƯU TẠI: {report_path}")
    print("="*60)

if __name__ == "__main__":
    train_lgbm()