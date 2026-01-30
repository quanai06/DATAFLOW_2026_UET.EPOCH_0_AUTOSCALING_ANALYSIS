import os
import pandas as pd

# Hàm này nạp file CSV kết quả dự báo mà bạn đã chạy từ XGBoost và hợp nhất chúng lại
# Chuyển về dạng req/s và bytes/s cho từng khoảng thời gian
# Xử lý khoảng trống bằng forward-fill

def change_to_per_second(model, timeframe, suffix='' ):
    df_req = pd.read_csv(os.path.join('results', model, f'results_{model}_y_req_t1_{timeframe}{suffix}.csv'))
    df_bytes = pd.read_csv(os.path.join('results', model, f'results_{model}_y_bytes_imp_t1_{timeframe}{suffix}.csv'))
    
    # Chuyển đổi dự báo về đơn vị per second
    factor = {'1m': 60, '5m': 300, '15m': 900}[timeframe]
    
    df_req['y_req_t1'] = df_req['y_req_t1'] / factor
    df_bytes['y_bytes_imp_t1'] = df_bytes['y_bytes_imp_t1'] / factor
    df_req['predicted'] = df_req['predicted'] / factor
    df_bytes['predicted'] = df_bytes['predicted'] / factor
    
    df_req = df_req.rename(columns={'y_req_t1': f'act_{timeframe}{suffix}_req', 'predicted': f'predicted_{timeframe}{suffix}_req'})
    df_bytes = df_bytes.rename(columns={'y_bytes_imp_t1': f'act_{timeframe}{suffix}_bytes', 'predicted': f'predicted_{timeframe}{suffix}_bytes'})
    df_req['timestamp'] = pd.to_datetime(df_req['timestamp'])
    df_bytes['timestamp'] = pd.to_datetime(df_bytes['timestamp'])
    return df_req, df_bytes

def merge_multiresolution_data(model):
    print(">>> Đang hợp nhất dữ liệu đa độ phân giải...")
    
    # Load và đổi đơn vị
    req_1m, bytes_1m = change_to_per_second(model, '1m')
    req_1m_q90, bytes_1m_q90 = change_to_per_second(model, '1m', suffix='_q90')
    req_5m, bytes_5m = change_to_per_second(model, '5m')
    req_5m_q90, bytes_5m_q90 = change_to_per_second(model, '5m', suffix='_q90')
    req_15m, bytes_15m = change_to_per_second(model, '15m')
    
    # Merge
    df = pd.merge(req_1m, bytes_1m, on='timestamp')
    
    df = pd.merge(df, req_1m_q90[['timestamp', 'predicted_1m_q90_req']], on='timestamp', how='left')
    df = pd.merge(df, bytes_1m_q90[['timestamp', 'predicted_1m_q90_bytes']], on='timestamp', how='left')
    
    df = pd.merge(df, req_5m[['timestamp', 'predicted_5m_req']], on='timestamp', how='left')
    df = pd.merge(df, bytes_5m[['timestamp', 'predicted_5m_bytes']], on='timestamp', how='left')

    df = pd.merge(df, req_5m_q90[['timestamp', 'predicted_5m_q90_req']], on='timestamp', how='left')
    df = pd.merge(df, bytes_5m_q90[['timestamp', 'predicted_5m_q90_bytes']], on='timestamp', how='left')

    df = pd.merge(df, req_15m[['timestamp', 'predicted_15m_req']], on='timestamp', how='left')
    df = pd.merge(df, bytes_15m[['timestamp', 'predicted_15m_bytes']], on='timestamp', how='left')
    
    # Xử lý khoảng trống bằng forward-fill
    df = df.sort_values('timestamp')
    df = df.ffill()
    df = df.dropna().reset_index(drop=True)

    return df

if __name__ == "__main__":
    # df = merge_multiresolution_data('xgboost')
    # df.to_csv('results/merged_xgboost_data.csv', index=False)
    # print("Đã lưu dữ liệu hợp nhất tại: results/merged_xgboost_data.csv")

    df = merge_multiresolution_data('lgbm')
    df.to_csv('results/merged_lgbm_data.csv', index=False)
    print("Đã lưu dữ liệu hợp nhất tại: results/merged_lgbm_data.csv")
