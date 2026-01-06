
import torch
import torch.nn as nn

FUTURE = 60


class CNN_LSTM_MultiHeadAttention(nn.Module):
    def __init__(self, input_features=14, hidden_dim=256, 
                 num_layers=3, output_steps=FUTURE, dropout_rate=0.2, num_heads=4):
        super(CNN_LSTM_MultiHeadAttention, self).__init__()
        
        self.conv1 = nn.Conv1d(input_features, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2, 2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2, 2)
        
        self.lstm = nn.LSTM(128, hidden_dim, num_layers=num_layers, batch_first=True,
                            dropout=dropout_rate if num_layers > 1 else 0, bidirectional=True)
        lstm_output_dim = hidden_dim * 2 
        

        self.attention = nn.MultiheadAttention(embed_dim=lstm_output_dim, 
                                               num_heads=num_heads, 
                                               dropout=dropout_rate,
                                               batch_first=True) # Quan trọng: batch_first=True
        

        self.fc1 = nn.Linear(lstm_output_dim, 256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_steps)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.to(torch.float32) # (B,L,F)
        
        x = x.transpose(1, 2) # (B, F, L)
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        x = x.transpose(1, 2) # (B, L', 64)
        
        lstm_out, (hn, cn) = self.lstm(x)
        #lstm_out = [B,L',512]
        
        query = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1)
        query = query.unsqueeze(1) 
        key = lstm_out
        value = lstm_out
        
        attn_output, attn_weights = self.attention(query, key, value)
        x = attn_output.squeeze(1)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        
        return x