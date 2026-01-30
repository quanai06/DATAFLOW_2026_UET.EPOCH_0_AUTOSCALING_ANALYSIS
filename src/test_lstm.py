import pandas as pd
import numpy as np
import os
import joblib
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1. Đăng ký lại hàm loss tùy chỉnh để Keras có thể nhận diện khi load
@tf.keras.utils.register_keras_serializable()
def pinball_loss_09(y_true, y_pred):
    quantile = 0.9
    error = y_true - y_pred
    return tf.reduce_mean(tf.maximum(quantile * error, (quantile - 1) * error))

def prepare_inference_data(df_train_tail, df_test, features, target_col, scaler_x, scaler_y, window_size):
    """
    Chuẩn bị dữ liệu cho inference bằng cách ghép đuôi tập train để làm window cho tập test.
    """
    # Kết hợp dữ liệu để đảm bảo không mất các dòng đầu của tập test
    combined = pd.concat([df_train_tail, df_test], axis=0).reset_index(drop=True)
    
    # Scale dữ liệu
    x_scaled = scaler_x.transform(combined[features])
    y_scaled = scaler_y.transform(combined[[target_col]])
    
    xs, ys = [], []
    for i in range(len(x_scaled) - window_size):
        xs.append(x_scaled[i : (i + window_size)])
        ys.append(y_scaled[i + window_size])
        
    return np.array(xs), np.array(ys)

def run_test_inference(run_folder, timeframe):
    """
    Hàm gọi model và thực hiện test.
    run_folder: Đường dẫn đến folder cụ thể của model 
    """
    print(f"\n--- Đang thực hiện Inference cho: {run_folder} ---")
    
    # 1. Load Scalers và Model
    scaler_x = joblib.load(f"{run_folder}/scaler_x.pkl")
    scaler_y = joblib.load(f"{run_folder}/scaler_y.pkl")
    
    # Load model với custom_objects cho pinball_loss
    model = tf.keras.models.load_model(
        f"{run_folder}/model.keras", 
        custom_objects={'pinball_loss_09': pinball_loss_09}
    )
    
    # Load Metadata để lấy window_size và target_col
    import json
    with open(f"{run_folder}/metadata.json", "r") as f:
        meta = json.load(f)
    
    target_col = meta['target']
    features = meta['features']
    window_size = meta['best_params']['window_size']
    
    # 2. Load dữ liệu thực tế từ disk
    train_path = f'data/model_dl/train_{timeframe}.parquet'
    test_path = f'data/model_dl/test_{timeframe}.parquet'
    
    df_train = pd.read_parquet(train_path).sort_values('timestamp')
    df_test = pd.read_parquet(test_path).sort_values('timestamp')
    
    # 3. Chuẩn bị dữ liệu 3D cho LSTM
    X_test_seq, y_test_seq = prepare_inference_data(
        df_train.tail(window_size), 
        df_test, 
        features, 
        target_col, 
        scaler_x, 
        scaler_y, 
        window_size
    )
    
    # 4. Dự báo
    preds_scaled = model.predict(X_test_seq, verbose=0)
    preds = scaler_y.inverse_transform(preds_scaled).flatten()
    actual = scaler_y.inverse_transform(y_test_seq).flatten()
    
    # 5. Tính toán metrics trên tập Test
    rmse = np.sqrt(mean_squared_error(actual, preds))
    mae = mean_absolute_error(actual, preds)
    
    print(f"SỐ LIỆU TẬP TEST:")
    print(f" - RMSE: {rmse:.4f}")
    print(f" - MAE:  {mae:.4f}")
    
    # 6. Lưu kết quả ra file riêng biệt
    results_df = pd.DataFrame({
        'timestamp': df_test['timestamp'].values,
        'actual': actual,
        'predicted': preds
    })
    
    output_path = f"results/lstm/results_lstm_test_{target_col}_{timeframe}.csv"
    results_df.to_csv(output_path, index=False)
    print(f"✅ Đã lưu kết quả tại: {output_path}")

if __name__ == "__main__":
    timeframes = ['1m', '5m', '15m']
    targets = ['y_req_t1', 'y_bytes_imp_t1']
    
    for t_f in timeframes:
        for target in targets:
            MY_RUN_FOLDER = f"models/lstm/lstm_{target}_{t_f}" 
            run_test_inference(MY_RUN_FOLDER, timeframe=t_f)