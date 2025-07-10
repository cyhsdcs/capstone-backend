import warnings
warnings.filterwarnings("ignore")
import os
import pandas as pd
from .model_builder import TransformerModelBuilder
from .data_loader import DataLoader
from .preprocessing import DataPreprocessor
from .feature_engineer import FeatureEngineer
from .config import Config
from keras.optimizers import Adam

def predict_by_Transformer(pid: str, iflstm: bool = False):
    pid = 'product'+pid
    # pid 保持字符串格式，不需要转换为int
    loader = DataLoader(
        product_path='data/product_info_simple_final_train.csv',
        yield_path='data/cbyieldcurve_info_final.csv',
        time_path='data/time_info_final.csv',
        predict_path='data/predict_table.csv'
    )
    if iflstm == True:
        model_type = 'LSTM'
    else:
        model_type = 'Transformer'
    saved_dir = os.path.join('model', model_type)


    loader.convert_datetime()
    loader.drop_nat()
    loader.fill_missing_values()
    loader.align_train_test_pids()
    loader.merge_data()
    df_product, df_predict = loader.get_product_data()


    fe = FeatureEngineer(df_product)
    fe.create_time_features()
    fe.add_normalized_time_feature()  # 调用新增的时间标准化特征方法
    fe.scale_uv_features()
    fe.log_transform_yield()
    fe.create_lag_features()  # 包含 dropna
    fe.fill_missing_mean()
    product_data = fe.get_df()


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

    train_df_balanced = train_df.copy()

    # PID编码
    train_df_balanced = dp.encode_pid(train_df_balanced)
    
    model_builder = TransformerModelBuilder()
    pid_vocab_size = len(dp.le.classes_)
    # 1. 创建新的模型实例（与训练时相同的配置）
    if iflstm == True:
        inference_model = model_builder.build_lstm_attention_model_with_pid(
            input_shape=(10, 23),
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
            input_shape=(10, 23), 
            pid_vocab_size=pid_vocab_size, 
            forecast_horizon=Config.OUTPUT_WINDOW, 
            target_len=len(Config.TARGET)
        )

    # 2. 编译模型（必须与训练时相同）
    inference_model.compile(optimizer=Adam(learning_rate=1e-4, clipvalue=1.0), 
                            loss='mae', 
                            metrics=[model_builder.wmape, 'mae', 'mse'])

    checkpoint_path = os.path.join(saved_dir, 'best_model.weights.h5')
    inference_model.load_weights(checkpoint_path)
    print(f"已加载模型权重: {checkpoint_path}")

    X_test, test_pids_encoded = dp.prepare_test_set_with_pid_single(product_data, window_size=Config.INPUT_WINDOW, pid=pid)
    y_pred_test_scaled = inference_model.predict([X_test, test_pids_encoded])
    y_pred_test = dp.target_scaler.inverse_transform(y_pred_test_scaled.reshape(-1, y_pred_test_scaled.shape[-1]))

    apply_amt_pred = y_pred_test[:,0]
    redeem_amt_pred = y_pred_test[:,1]
    net_in_amt_pred = y_pred_test[:,2]
    net_in_amt_pred_transformer = apply_amt_pred - redeem_amt_pred

    # 生成未来10个交易日的日期
    last_10_days = product_data[product_data['product_pid'] == pid].tail(10)
    last_date_raw = last_10_days['transaction_date'].iloc[-1]
    # 统一格式为YYYYMMDD字符串
    if hasattr(last_date_raw, 'strftime'):
        last_date_str = last_date_raw.strftime('%Y%m%d')
    else:
        last_date_str = str(last_date_raw)
    
    # 读取交易日信息
    time_info = pd.read_csv('data/time_info_final.csv')
    time_info['stat_date'] = time_info['stat_date'].astype(str)
    
    # 找到最后一个交易日在time_info中的位置
    matching_dates = time_info[time_info['stat_date'] == last_date_str]
    if len(matching_dates) > 0:
        last_date_idx = matching_dates.index[0]
    else:
        # 如果找不到对应日期，使用time_info的最后一个交易日
        last_date_idx = time_info[time_info['is_trade'] == 1].index[-1]
        print(f"Warning: {last_date_str} not found in time_info, using last available trading date")
    
    # 获取未来10个交易日
    future_trading_dates = []
    current_idx = last_date_idx + 1
    count = 0
    
    while count < 10 and current_idx < len(time_info):
        if time_info.iloc[current_idx]['is_trade'] == 1:
            future_trading_dates.append(time_info.iloc[current_idx]['stat_date'])
            count += 1
        current_idx += 1

    # 构建返回结果
    result = {
        "time_list": future_trading_dates,
        "apply_amt_pred": apply_amt_pred.tolist(),
        "redeem_amt_pred": redeem_amt_pred.tolist(),
        "net_in_amt_pred": (net_in_amt_pred * 0.5 + net_in_amt_pred_transformer * 0.5).tolist()
    }

    return result   


if __name__ == "__main__":
    pid = "product1"  # 改为字符串格式
    iflstm = False
    df_predict = predict_by_Transformer(pid, iflstm)
    print(df_predict)


