
import requests 
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import glob
import time
import ccxt
import joblib                  
import os 
from datetime import datetime 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import mean_squared_error, mean_absolute_error 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

SYMBOL = "ETHUSDT"        
INTERVAL = "5m"          
LIMIT = 3000            
LOOKBACK = 100  
FUTURE = 60
TEST_RATIO = 0.15
VAL_RATIO = 0.2
EPOCHS = 100
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)