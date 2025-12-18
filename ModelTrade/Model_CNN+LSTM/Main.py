def main():
  X_train, y_train, X_val, y_val, X_test, y_test, scaler, close_prices_raw,test_data,test_one,test_with_history ,train_val_data= load_data()
  test_preds, test_actual = train_model(X_train, y_train, X_val, y_val, X_test, y_test, EPOCHS,test_data, test_one,test_with_history)
  plot_test_comparison2( test_actual, test_preds, title="So sánh Dự đoán và Thực tế (Train + Val + Test)")

if __name__ == "__main__":  
    main()