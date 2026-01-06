import matplotlib.pyplot as plt 

def plot_test_comparison2(test_actual, test_preds, title="So sánh Dự đoán và Thực tế (Chỉ tập Test)"):
    """
    Vẽ biểu đồ so sánh CHỈ giá trị thực tế và dự đoán của tập Test.
    
    Parameters:
    - test_actual: Array chứa giá thực tế của test (Ground Truth)
    - test_preds: Array chứa giá dự đoán của test (Prediction)
    """
    
    # 1. Chuẩn bị dữ liệu (Flatten để đảm bảo là mảng 1 chiều)
    # Chuyển về numpy array phòng trường hợp đầu vào là list hoặc tensor
    y_true = np.array(test_actual).flatten()
    y_pred = np.array(test_preds).flatten()
    
    # Tạo index (trục thời gian) bắt đầu từ 0 đến hết tập test
    steps = np.arange(len(y_true))
    
    # 2. Khởi tạo Biểu đồ
    plt.figure(figsize=(14, 7))
    
    # 3. Vẽ đường Thực tế (Actual)
    plt.plot(steps, y_true, label='Giá trị Thực tế (Actual)', 
             color='#007acc', linewidth=2.5, marker='o', markersize=5, alpha=0.9)
    
    # 4. Vẽ đường Dự đoán (Prediction)
    plt.plot(steps, y_pred, label='Giá trị Dự đoán (Prediction)', 
             color='#ff4d4d', linewidth=2, linestyle='--', marker='x', markersize=6, alpha=0.9)
    
    # 5. Tô màu vùng sai số (Error Gap)
    # Giúp nhìn rõ đoạn nào mô hình dự đoán sai nhiều nhất
    plt.fill_between(steps, y_true, y_pred, color='gray', alpha=0.2, label='Biên độ sai số (Error Gap)')
    
    # 6. Vẽ đường xu hướng (Trendlines) - Tùy chọn để xem mô hình có bắt được hướng đi không
    if len(y_true) > 1:
        # Xu hướng thực tế
        z_true = np.polyfit(steps, y_true, 1)
        p_true = np.poly1d(z_true)
        plt.plot(steps, p_true(steps), color='blue', linestyle=':', alpha=0.5, linewidth=1)
        
        # Xu hướng dự đoán
        z_pred = np.polyfit(steps, y_pred, 1)
        p_pred = np.poly1d(z_pred)
        plt.plot(steps, p_pred(steps), color='red', linestyle=':', alpha=0.5, linewidth=1)

    # 7. Formatting (Trang trí)
    plt.title(title, fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Các bước thời gian (Time Steps)', fontsize=12)
    plt.ylabel('Giá (Price)', fontsize=12)
    plt.legend(loc='best', fontsize=11, frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Đánh dấu trục X là số nguyên
    plt.xticks(steps, rotation=45 if len(steps) > 20 else 0)
    
    plt.tight_layout()
    plt.show()
    
    # 8. In thống kê chi tiết
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"\n{'='*50}")
    print(f"📊 THỐNG KÊ KẾT QUẢ TEST")
    print(f"{'-'*50}")
    print(f"🔹 Số lượng mẫu test: {len(y_true)}")
    print(f"🔹 MAE (Sai số tuyệt đối trung bình): {mae:.4f}")
    print(f"🔹 RMSE (Căn bậc hai sai số bình phương): {rmse:.4f}")
    print(f"🔹 MAPE (Sai số phần trăm trung bình): {mape:.2f}%")
    
    # Tính độ chính xác hướng đi (Directional Accuracy)
    # So sánh dấu của (Giá[t] - Giá[t-1]) giữa thực tế và dự đoán
    if len(y_true) > 1:
        diff_true = np.diff(y_true)
        diff_pred = np.diff(y_pred)
        correct_direction = np.sum(np.sign(diff_true) == np.sign(diff_pred))
        direction_acc = (correct_direction / len(diff_true)) * 100
        print(f"🔹 Directional Accuracy (Đoán đúng xu hướng tăng/giảm): {direction_acc:.2f}%")
        
    print(f"{'='*50}\n")