def load_data():
    df = fetch_klines(SYMBOL, INTERVAL, LIMIT)
    close_prices_raw = df['close'].values.astype(np.float32)
    
    n_total = len(df)
    test_size = FUTURE
    test_size_all = int(n_total*TEST_RATIO)
    val_size = int(n_total * VAL_RATIO)
    train_size = n_total - test_size_all - val_size
    
    train_size = max(train_size, LOOKBACK + 100)
    test_size = FUTURE
    val_size = max(val_size, 50)
    
    train_raw = df[:train_size]
    val_raw = df[train_size: train_size + val_size]
    train_val_data = pd.concat([train_raw, val_raw])
    test_raw = df[train_size + val_size:train_size + val_size+FUTURE]

    #print(f" {len(test_raw)}")
    test_one = test_raw[:1] # giá thực tế của bước test đầu tiên
    test_one = test_one['close'].values.astype(np.float32)
    
    max_window = max(30, LOOKBACK)  # Max size của rolling
    train_df = add_features(train_raw)
    val_with_history = pd.concat([train_raw.iloc[-max_window:], val_raw])
    val_df = add_features(val_with_history).iloc[-val_size:]  
    

    test_with_history = pd.concat([df.iloc[train_size + val_size-LOOKBACK: train_size + val_size], test_raw])
    # tập test lúc này là tập test(30) + lookback
    
    test_df = add_features(test_with_history).iloc[:LOOKBACK+1] 

    #print(f" ssssss {test_df.shape}")
    # tập test gồm giá trị đầu của tập test và lookback
    test_df_fake = add_features(test_with_history).iloc[:LOOKBACK+1]
    
    # Lấy feats data
    feats = ["close", "volume", 'return_1','return_3','roc_10',"bb_width","atr_14_pct","dist_ma_20","dist_ma_50","rvol","rsi_14","macd_hist","dist_to_upper","pump_signal"]
    
    train_data = train_df[feats].values.astype(np.float32)
    val_data = val_df[feats].values.astype(np.float32)
    test_data = test_df[feats].values.astype(np.float32)
    # tập gồm 1 giá trị test và lookback
    test_data_fake = test_df_fake[feats].values.astype(np.float32)
    
    print(f"Train shape: {train_data.shape}, Val: {val_data.shape}, Test: {test_data.shape}")
    print(f"{test_data_fake.shape}")
    scaled_full, scaler = minmax(train_data, val_data, test_data_fake)
    X_all, y_all = create_sequences_logreturn(scaled_full, close_prices_raw, LOOKBACK)
    train_count = max(0, len(train_data) - LOOKBACK)
    val_count = max(0, len(val_data))
    
    X_train = X_all[:train_count]
    y_train = y_all[:train_count]
    X_val = X_all[train_count: train_count + val_count]
    y_val = y_all[train_count: train_count + val_count]
    X_test = X_all[train_count + val_count:]
    y_test = y_all[train_count + val_count:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, close_prices_raw,test_data,test_one,test_with_history,train_val_data