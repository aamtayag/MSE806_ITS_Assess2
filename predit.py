import os
import sqlite3
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

# Scikit-learn for linear regression
from sklearn.linear_model import LinearRegression  # 或者使用 SGDRegressor 实现在线训练
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# TensorFlow/Keras for LSTM model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Database file path
DB_PATH = 'E:/Yoobee/MSE806-ITS/AS2/SmartParking.db'
# 模型保存路径
LR_MODEL_PATH = 'lr_model.pkl'
LSTM_MODEL_PATH = 'lstm_model.h5'

def load_data(db_path):
    conn = sqlite3.connect(db_path)
    query = """
        SELECT 
            pd2.parking_lot_id,
            pd2.start_time,
            pl.total_spaces,
            pl.available_spaces,
            (pl.total_spaces - pl.available_spaces) AS occupancy
        FROM parkingdata2 pd2
        JOIN parkinglots pl ON pd2.parking_lot_id = pl.lot_id
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def preprocess_data(df):
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['timestamp'] = df['start_time'].apply(lambda x: x.timestamp())
    X = df[['parking_lot_id', 'timestamp']]
    y = df['occupancy']
    return X, y

def load_or_create_lr_model(model_path):
    if os.path.exists(model_path):
        print("加载已存在的线性回归模型...")
        with open(model_path, 'rb') as f:
            lr_model = pickle.load(f)
    else:
        print("创建新的线性回归模型...")
        lr_model = LinearRegression()
    return lr_model

def save_lr_model(lr_model, model_path):
    with open(model_path, 'wb') as f:
        pickle.dump(lr_model, f)
    print("线性回归模型已保存。")

def train_linear_regression(lr_model, X_train, y_train, X_test, y_test):
    # 注意：LinearRegression 不支持增量训练，如需在线更新，可考虑 SGDRegressor
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    mse = mean_squared_error(y_test, lr_preds)
    print("线性回归测试MSE:", mse)
    return lr_model

def load_or_create_lstm_model(model_path, input_shape):
    if os.path.exists(model_path):
        print("加载已存在的LSTM模型...")
        lstm_model = tf.keras.models.load_model(model_path)
    else:
        print("创建新的LSTM模型...")
        lstm_model = Sequential()
        lstm_model.add(LSTM(50, input_shape=input_shape))
        lstm_model.add(Dense(1))
        lstm_model.compile(loss='mse', optimizer='adam')
    return lstm_model

def save_lstm_model(lstm_model, model_path):
    lstm_model.save(model_path)
    print("LSTM模型已保存。")

def train_lstm(lstm_model, X_train, y_train, X_test, y_test, epochs=10, batch_size=64):
    # 重塑输入数据形状： [samples, time_steps, features]
    X_train_lstm = np.expand_dims(X_train.values, axis=1)
    X_test_lstm = np.expand_dims(X_test.values, axis=1)
    
    es = EarlyStopping(monitor='loss', patience=3, verbose=1)
    lstm_model.fit(X_train_lstm, y_train.values, epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=1)
    
    lstm_preds = lstm_model.predict(X_test_lstm)
    mse = mean_squared_error(y_test, lstm_preds)
    print("LSTM测试MSE:", mse)
    return lstm_model

def predict_capacity(lr_model, lstm_model, parking_lot_id, future_datetime_str):
    future_dt = datetime.strptime(future_datetime_str, '%Y-%m-%d %H:%M:%S')
    future_timestamp = future_dt.timestamp()
    X_new = np.array([[parking_lot_id, future_timestamp]])
    
    lr_prediction = lr_model.predict(X_new)[0]
    X_new_lstm = np.expand_dims(X_new, axis=1)
    lstm_prediction = lstm_model.predict(X_new_lstm)[0, 0]
    
    return lr_prediction, lstm_prediction

def store_predictions(db_path, parking_lot_id, prediction_datetime, lr_prediction, lstm_prediction):
    created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    insert_query = """
        INSERT INTO ai_predictions (parking_lot_id, prediction_datetime, model, predicted_value, created_at)
        VALUES (?, ?, ?, ?, ?)
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(insert_query, (parking_lot_id, prediction_datetime, 'LinearRegression', lr_prediction, created_at))
    cur.execute(insert_query, (parking_lot_id, prediction_datetime, 'LSTM', lstm_prediction, created_at))
    conn.commit()
    conn.close()
    print("预测结果已成功存入数据库。")

def main():
    print("加载和预处理数据...")
    df = load_data(DB_PATH)
    X, y = preprocess_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 线性回归模型部分
    lr_model = load_or_create_lr_model(LR_MODEL_PATH)
    lr_model = train_linear_regression(lr_model, X_train, y_train, X_test, y_test)
    save_lr_model(lr_model, LR_MODEL_PATH)
    
    # LSTM模型部分
    # 注意输入形状：这里是 (time_steps, features)，time_steps=1
    input_shape = (1, X_train.shape[1])
    lstm_model = load_or_create_lstm_model(LSTM_MODEL_PATH, input_shape)
    lstm_model = train_lstm(lstm_model, X_train, y_train, X_test, y_test)
    save_lstm_model(lstm_model, LSTM_MODEL_PATH)
    
    # 示例预测
    parking_lot_id = 1
    future_datetime = '2025-03-15 12:00:00'
    lr_pred, lstm_pred = predict_capacity(lr_model, lstm_model, parking_lot_id, future_datetime)
    
    print("预测停车占用 (线性回归):", lr_pred)
    print("预测停车占用 (LSTM):", lstm_pred)
    
    store_predictions(DB_PATH, parking_lot_id, future_datetime, lr_pred, lstm_pred)

def test():
    lr_model = load_or_create_lr_model(LR_MODEL_PATH)
    # 加载保存的 LSTM 模型，注意 input_shape 要与训练时保持一致（这里是 (1, 特征数)）
    input_shape = (1, 2)  # 特征为 parking_lot_id 和 timestamp
    lstm_model = load_or_create_lstm_model(LSTM_MODEL_PATH, input_shape)
    parking_lot_id = 1
    future_datetime = '2025-03-19 11:00:00'
    
    # 调用预测函数
    lr_prediction, lstm_prediction = predict_capacity(lr_model, lstm_model, parking_lot_id, future_datetime)

    print("预测停车占用 (线性回归):", lr_prediction)
    print("预测停车占用 (LSTM):", lstm_prediction)

if __name__ == '__main__':
    #main()
    test()
