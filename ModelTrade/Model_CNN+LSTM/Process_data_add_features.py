    def fetch_klines(symbol=SYMBOL, interval=INTERVAL, limit=LIMIT):
        exchange = ccxt.binance()
        
       
        duration_in_seconds = exchange.parse_timeframe(interval)
        duration_in_ms = duration_in_seconds * 1000
        
        # Tính thời điểm bắt đầu: Hiện tại - (Số nến cần lấy * Thời gian 1 nến)
        # Cộng thêm buffer (ví dụ 100 nến) để trừ hao các khoảng thời gian sàn bảo trì hoặc mất dữ liệu
        now = exchange.milliseconds()
        since = now - (limit * duration_in_ms) - (100 * duration_in_ms)
        
        all_ohlcv = []
        
        # 2. Vòng lặp lấy dữ liệu (Pagination)
        while len(all_ohlcv) < limit:
            try:
                current_limit = LIMIT 
                ohlcv = exchange.fetch_ohlcv(symbol, interval, since=since, limit=current_limit)
                
                if not ohlcv:
                    break 
                
                all_ohlcv.extend(ohlcv)
                last_timestamp = ohlcv[-1][0]
                since = last_timestamp + 1
                
                # Nếu đã lấy đến hiện tại thì dừng
                if since > now:
                    break
                    
                # Thêm delay nhỏ để tránh bị sàn chặn (Rate Limit)
                time.sleep(0.1) 
                
            except Exception as e:
                print(f"Lỗi khi tải dữ liệu: {e}")
                break

        #Chuyển đổi sang DataFrame
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Xử lý dữ liệu
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
        
        df = df.drop_duplicates(subset=['timestamp'])
        if len(df) > limit:
            df = df.iloc[-limit:].reset_index(drop=True)
            
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        
        print(f"Đã tải {len(df)} nến cho {symbol}")
        return df
  
       # ---------- 1.2) Tạo các chỉ số kỹ thuật ----------
    def add_features(data):
        # 1. Chuyển đổi dữ liệu
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        # Đảm bảo tên cột chuẩn (nếu dữ liệu gốc viết hoa)
        df.columns = [col.lower() for col in df.columns] 
    
        # --- NHÓM 1: MOMENTUM (Xung lượng - Giữ nguyên cái tốt) ---
        df["return_1"] = df["close"].pct_change(1)   # Biến động 1 phiên
        df["return_3"] = df["close"].pct_change(3)   # Biến động 3 phiên
        df["roc_10"] = df["close"].pct_change(10)    # Rate of Change
        
        # --- NHÓM 2: VOLATILITY (Bắt buộc để bắt "độ giật") ---
        # a. Bollinger Bands Width (Báo hiệu sắp nổ Volatility)
        rolling_mean_20 = df["close"].rolling(window=20).mean()
        rolling_std_20 = df["close"].rolling(window=20).std()
        upper_band = rolling_mean_20 + (2 * rolling_std_20)
        lower_band = rolling_mean_20 - (2 * rolling_std_20)
        
        # Feature 1: Độ rộng dải băng (Co thắt = Sắp biến động mạnh)
        df["bb_width"] = (upper_band - lower_band) / rolling_mean_20
        
        # b. ATR (Average True Range) - Đo biên độ nến thực tế
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        
        # Lấy max của 3 giá trị trên cho từng dòng
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        # Feature 2: ATR chuẩn hóa theo giá (để không bị phụ thuộc vào mức giá 60k hay 20k)
        df["atr_14_pct"] = true_range.rolling(window=14).mean() / df["close"]
    
        # --- NHÓM 3: TREND DISTANCE (Thay thế MA thô) ---
        # Thay vì đưa giá ma_5 (vd: 50000), ta đưa khoảng cách % từ giá đến MA
        # Giúp model hiểu giá đang "quá căng" (Overextended) hay đang "hồi về" (Mean Reversion)
        df["dist_ma_20"] = (df["close"] - rolling_mean_20) / rolling_mean_20
        df["dist_ma_50"] = (df["close"] - df["close"].rolling(window=50).mean()) / df["close"].rolling(window=50).mean()
    
        # --- NHÓM 4: VOLUME FLOW (Xăng cho xe chạy) ---
        # Feature 3: RVOL (Relative Volume) - Volume hiện tại gấp mấy lần trung bình?
        # Nếu dữ liệu không có volume, hãy comment dòng này lại
        if 'volume' in df.columns:
            vol_ma_20 = df["volume"].rolling(window=20).mean()
            # Cộng 1e-6 để tránh chia cho 0
            df["rvol"] = df["volume"] / (vol_ma_20 + 1e-6)
        else:
            df["rvol"] = 0 # Giá trị mặc định nếu không có volume
    
        # --- NHÓM 5: INDICATORS KHÁC ---
        df["rsi_14"] = compute_rsi(df["close"], 14) / 100.0 # Chuẩn hóa về 0-1 ngay lập tức
    
        # MACD: Lấy Histogram (quan trọng hơn đường tín hiệu để bắt đảo chiều sớm)
        ema_12 = df["close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["close"].ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9, adjust=False).mean()
        
        # Feature 4: MACD Histogram
        df["macd_hist"] = macd - signal

        #======================================================================================================
        # 1. Tính toán Bollinger Bands cơ bản
        rolling_mean = df["close"].rolling(window=20).mean()
        rolling_std = df["close"].rolling(window=20).std()
        upper_band = rolling_mean + (2 * rolling_std)
        lower_band = rolling_mean - (2 * rolling_std)
        
        # --- CÁC FEATURES MỚI TỪ UPPER BAND ---
        
        # Feature 1: Khoảng cách % đến dải trên (Quan trọng nhất)
        # Dương = Đang Pump, Âm = Dưới kháng cự
        df["dist_to_upper"] = (df["close"] - upper_band) / upper_band
    
        
        # Feature 3: Breakout với Volume (Interaction Feature)
        # Ý tưởng: Giá vượt dải trên VÀ Volume lớn => Cú giật cực mạnh sắp tới
        # rvol là Relative Volume bạn đã tính trước đó
        # Đây là feature "nhân tạo" giúp model học nhanh hơn
        is_breakout = (df["close"] > upper_band).astype(float)
        df["pump_signal"] = is_breakout * df["rvol"]
        #==============================================================================================
        df = df.dropna() 
        
        # Thay thế inf nếu có (do chia cho 0)
        df = df.replace([np.inf, -np.inf], 0)
    
        # Chọn lọc Feature cuối cùng để đưa vào Model (Quan trọng!)
        # Loại bỏ 'open', 'high', 'low', 'close', 'volume' gốc và các biến trung gian
        # Chỉ giữ lại các Features đã tính toán (Stationary Features)

        
        return df


    def compute_rsi(series, period=14):
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ma_up = up.ewm(com=period-1, adjust=True).mean()
        ma_down = down.ewm(com=period-1, adjust=True).mean()
        rs = ma_up / (ma_down + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi