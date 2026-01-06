import pandas as pd
import numpy as np

def train_model(X_train, y_train, X_val, y_val, X_test, y_test, EPOCHS,test_data, test_one,test_with_history):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    # Convert to tensors
    X_train_t = torch.from_numpy(X_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).float().to(device)  
    X_val_t = torch.from_numpy(X_val).float().to(device)
    y_val_t = torch.from_numpy(y_val).float().to(device)  
    X_test_t = torch.from_numpy(X_test).float().to(device)
    y_test_t = torch.from_numpy(y_test).float().to(device) 
    
    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # ==================== Initialize Model ====================
    model = CNN_LSTM_MultiHeadAttention(
        input_features=14,
        hidden_dim=128,
        num_layers=2,
        output_steps=FUTURE,  
        dropout_rate=0.2
    ).to(device)
#==================================================================================================
    criterion = nn.MSELoss()

    current_lr = 0.9  
    sigma = 0.5
    kappa = 0.75

    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 3

    print("Start training CNN-LSTM with GDA Update...")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            current_lr, batch_loss_val = GDA_update(
                model,criterion, batch_x, batch_y, lr=current_lr, sigma=sigma, kappa=kappa)
            
            train_loss += batch_loss_val * batch_x.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # --- Validation ---
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
            val_loss = criterion(val_preds, y_val_t).item()
        
        # In thông tin bao gồm cả Learning Rate hiện tại để theo dõi
        print(f"Epoch {epoch}/{EPOCHS} | LR: {current_lr:.6f} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # --- Early stopping + save best ---
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_coin_cnn_lstm.pth"))
            patience_counter = 0
            print(f"  → New best model saved! Val loss: {val_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print("Early stopping triggered.")
                break
    
        
        # ======================================= Testing ===========================================

    
    print("Loading best model for walk-forward testing...")
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_coin_cnn_lstm.pth")))
    model.eval()
    device = next(model.parameters()).device
    
        # Dự đoán 30 bước tiếp theo từ bước đầu tiên của test
    print(f"1")
    with torch.no_grad():
        # 1. Chuyển sang tensor và đưa vào GPU
        X_first = torch.from_numpy(test_data[:-1]).float().to(device)
        # print(f"{X_first.shape}") # torch.Size([200, 12])
        X_first = X_first.unsqueeze(0) 

        # print(f"{X_first.shape}") #  torch.Size([1,200, 12])
        # 3. Dự đoán
        pred_logr = model(X_first).cpu().numpy().reshape(-1)
        initial_price = test_one # test_one là giá close của bước test đầu tiên 
        pred_prices = initial_price * np.cumprod(np.exp(pred_logr))
        
        #print(f"{pred_prices}")
        

    actuals_future = test_with_history.iloc[-FUTURE:]['close'].values
    
    print(f"Initial price: {initial_price}")
    print(f"Predicted next {FUTURE} prices: {pred_prices}")
    print(f"Actual next {FUTURE} prices: {actuals_future}")
    return pred_prices, actuals_future