def create_sequences_logreturn(scaled_data, close_raw, lookback=LOOKBACK, future_steps=FUTURE):
    X, y_logreturn = [], []
    close = np.asarray(close_raw, dtype=np.float32).flatten()
    print(f"độ dài close_raw = {len(close_raw)}")
    print(f"độ dài scaled_data = {len(scaled_data)}")
    
    for i in range(lookback, len(scaled_data) - future_steps + 1):
        X.append(scaled_data[i-lookback:i])
        
        # Tính log-returns cho future_steps ngày
        future_returns = []
        for j in range(1, future_steps + 1):
            ret = np.log(close[i+j-1] / close[i+j-2])
            future_returns.append(ret)
        
        y_logreturn.append(future_returns)
    
    return np.array(X, dtype=np.float32), np.array(y_logreturn, dtype=np.float32)


    
def minmax(train_data, val_data, test_data):  # Đưa dữ liệu từ giá thực về 0-1
    scaler = MinMaxScaler()
    scaler.fit(train_data)

    train_scaled = scaler.transform(train_data)
    val_scaled   = scaler.transform(val_data)
    test_scaled  = scaler.transform(test_data)

    scaled_full = np.vstack([train_scaled, val_scaled, test_scaled])
    return scaled_full, scaler

def inverse_close(scaled_values, scaler):
    scaled_values = np.asarray(scaled_values)
    inv=scaler.inverse_transform(scaled_values)
    return inv