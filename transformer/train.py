# train.py
import warnings
warnings.filterwarnings("ignore")
import os
import datetime
from config import Config
from data_loader import DataLoader
from feature_engineer import FeatureEngineer
from preprocessing import DataPreprocessor
from model_builder import TransformerModelBuilder
from keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
from keras.optimizers import Adam
import json
import numpy as np

# 新增导入Evaluator
from Evaluate import Evaluator
# print('process')
# 数据加载与初步处理
loader = DataLoader(
    product_path='data/product_info_simple_final_train.csv',
    yield_path='data/cbyieldcurve_info_final.csv',
    time_path='data/time_info_final.csv',
    predict_path='data/predict_table.csv'
)
print('process')
if Config.IFLSTM == True:
    model_type='LSTM'
else:
    model_type='Transformer'
saved_dir = os.path.join('saved_dir', model_type,datetime.datetime.now().strftime('%H%M%S'))
os.makedirs(saved_dir, exist_ok=True)
print('saved_path:', saved_dir)


loader.convert_datetime()
loader.drop_nat()
loader.fill_missing_values()
loader.align_train_test_pids()
loader.merge_data()
df_product, df_predict = loader.get_product_data()

# # 特征工程
fe = FeatureEngineer(df_product)
fe.create_time_features()
fe.add_normalized_time_feature()  # 调用新增的时间标准化特征方法
fe.scale_uv_features()
fe.log_transform_yield()
fe.create_lag_features()  # 包含 dropna
fe.fill_missing_mean()
product_data = fe.get_df()

# 数据预处理
dp = DataPreprocessor(
    df=product_data,
    df_predict=df_predict,
    features_to_scale=Config.FEATURES_TO_SCALE,
    target=Config.TARGET,
    train_start_date=Config.TRAIN_START_DATE,
    train_end_date=Config.TRAIN_END_DATE
)

train_df = dp.split_train_data()

# 拟合缩放器
dp.fit_scalers(train_df)

# 异常值处理
# train_df_balanced = dp.handle_outliers_iqr(train_df.copy(), Config.TARGET)
train_df_balanced = train_df.copy()

# PID编码
train_df_balanced = dp.encode_pid(train_df_balanced)

# 划分训练与验证集
X_train, y_train, X_val, y_val, train_pids_encoded, val_pids_encoded = dp.split_train_val(
    train_df_balanced, 
    validation_days=Config.VALIDATION_DAYS, 
    window_size=Config.INPUT_WINDOW, 
    forecast_horizon=Config.OUTPUT_WINDOW
)

# 构建模型
model_builder = TransformerModelBuilder()
pid_vocab_size = len(dp.le.classes_)

if Config.IFLSTM == True:
    model = model_builder.build_lstm_attention_model_with_pid(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        pid_vocab_size=pid_vocab_size,
        forecast_horizon=Config.OUTPUT_WINDOW,
        target_len=len(Config.TARGET),
        pid_embedding_dim=10,
        lstm_units=64,
        attention_units=64,
        dropout=0.1
    )
else:
    model = model_builder.build_transformer_model_with_pid(
        input_shape=(X_train.shape[1], X_train.shape[2]), 
        pid_vocab_size=pid_vocab_size, 
        forecast_horizon=Config.OUTPUT_WINDOW, 
        target_len=len(Config.TARGET)
    )

optimizer = Adam(learning_rate=1e-4, clipvalue=1.0)
model.compile(optimizer=optimizer, loss='mae', metrics=[model_builder.wmape, 'mae', 'mse'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
ckpt_callback = ModelCheckpoint(
    filepath=os.path.join(saved_dir, 'best_model.weights.h5'),
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=True,
    verbose=1
)
history = model.fit(
    [X_train, train_pids_encoded],
    y_train,
    epochs=100,
    batch_size=Config.BATCH_SIZE,
    validation_data=([X_val, val_pids_encoded], y_val),
    callbacks=[early_stopping, lr_scheduler,ckpt_callback]
)

# 使用Evaluator绘制训练过程
Evaluator.plot_history(history, save_path=os.path.join(saved_dir, 'loss.png'))

val_metrics = model.evaluate([X_val, val_pids_encoded], y_val)
print(f"验证集 - MAE: {val_metrics[2]}")
print(f"验证集 - (loss) WMAPE: {val_metrics[1]}")
print(f"验证集 - MSE: {val_metrics[3]}")


# 1. 创建新的模型实例（与训练时相同的配置）
if Config.IFLSTM == True:
    inference_model = model_builder.build_lstm_attention_model_with_pid(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        pid_vocab_size=pid_vocab_size,
        forecast_horizon=Config.OUTPUT_WINDOW,
        target_len=len(Config.TARGET),
        pid_embedding_dim=10,
        lstm_units=64,
        attention_units=64,
        dropout=0.1
    )
else:
    inference_model = model_builder.build_transformer_model_with_pid(
        input_shape=(X_train.shape[1], X_train.shape[2]), 
        pid_vocab_size=pid_vocab_size, 
        forecast_horizon=Config.OUTPUT_WINDOW, 
        target_len=len(Config.TARGET)
    )

# 2. 编译模型（必须与训练时相同）
inference_model.compile(optimizer=Adam(learning_rate=1e-4, clipvalue=1.0), 
                        loss='mae', 
                        metrics=[model_builder.wmape, 'mae', 'mse'])

# 3. 加载保存的权重
choose_ckpt_before = True  # 是否选择在之前实验的checkpoint
mode=str(125200)            #实验编码
if choose_ckpt_before:
    checkpoint_path = os.path.join(saved_dir[:-7],mode, 'best_model.weights.h5')
else:
    checkpoint_path = os.path.join(saved_dir, 'best_model.weights.h5')
inference_model.load_weights(checkpoint_path)
print(f"已加载模型权重: {checkpoint_path}")

y_pred_val_scaled = inference_model.predict([X_val, val_pids_encoded])
y_pred_val = dp.target_scaler.inverse_transform(y_pred_val_scaled.reshape(-1, y_pred_val_scaled.shape[-1]))
y_val_original = dp.target_scaler.inverse_transform(y_val.reshape(-1, y_val.shape[-1]))

def wmape_metric(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-8)
def mae_metric(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred)) 
def mse_metric(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred)**2) 
wmape = wmape_metric(y_val_original, y_pred_val)
print(f"验证集反归一化后 WMAPE: {wmape}")
mae= mae_metric(y_val_original, y_pred_val)
print(f"验证集反归一化后 MAE: {mae}")
mse = mse_metric(y_val_original, y_pred_val)
print(f"验证集反归一化后 MSE: {mse}")


wmape_filename = f"WMAPE_{wmape:.4f}"  
with open(os.path.join(saved_dir,wmape_filename), 'w', encoding='utf-8') as f:
    f.write('{}')
mae_filename = f"MAE_{mae:.4f}"
with open(os.path.join(saved_dir,mae_filename), 'w', encoding='utf-8') as f:
    f.write('{}')   
mse_filename = f"MSE_{mse:.4f}"
with open(os.path.join(saved_dir,mse_filename), 'w', encoding='utf-8') as f:
    f.write('{}')   


print('y_val_original.shape',y_val_original.shape)
# 对单个目标列例如apply_amt进行可视化对比(这里假设目标列顺序与TARGET一致)
Evaluator.plot_predictions(y_val_original[:, 0], y_pred_val[:, 0], metric='apply_amt', save_path=os.path.join(saved_dir, 'apply_amt.png'))
Evaluator.plot_predictions(y_val_original[:, 1], y_pred_val[:, 1], metric='redeem_amt', save_path=os.path.join(saved_dir, 'redeem_amt.png'))
Evaluator.plot_predictions(y_val_original[:, 2], y_pred_val[:, 2], metric='net_in_amt', save_path=os.path.join(saved_dir, 'net_in_amt.png'))

# 准备测试集并预测

# 4. 准备测试集数据
X_test, test_pids_encoded = dp.prepare_test_set_with_pid(product_data, window_size=Config.INPUT_WINDOW, forecast_horizon=Config.OUTPUT_WINDOW)
y_pred_test_scaled = inference_model.predict([X_test, test_pids_encoded])
y_pred_test = dp.target_scaler.inverse_transform(y_pred_test_scaled.reshape(-1, y_pred_test_scaled.shape[-1]))

apply_amt_pred = y_pred_test[:,0]
redeem_amt_pred = y_pred_test[:,1]
net_in_amt_pred = y_pred_test[:,2]
net_in_amt_pred_transformer = apply_amt_pred - redeem_amt_pred

df_predict['apply_amt_pred'] = apply_amt_pred
df_predict['redeem_amt_pred'] = redeem_amt_pred
df_predict['net_in_amt_pred'] = net_in_amt_pred * 0.5 + net_in_amt_pred_transformer * 0.5


df_predict.to_csv('test_data_with_predictions.csv', index=False)
