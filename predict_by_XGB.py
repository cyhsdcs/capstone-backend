# -*- coding: utf-8 -*-
"""
简化的XGB模型预测脚本 - 基于最后10天数据预测未来10天
"""
import pandas as pd
import xgboost
import os
from datetime import datetime, timedelta

def predict_future_10_days(product_pid, model_path="model/XGB/xgb.json", 
                          data_path="data/product_info_simple_final_train.csv"):
    """
    使用保存的XGB模型预测单个product未来10天
    
    Args:
        product_pid: 产品ID
        model_path: 模型文件路径
        data_path: 数据文件路径
        
    Returns:
        预测结果字典，包含未来10天的预测
    """
    # 1. 加载模型
    model = xgboost.XGBRegressor()
    model.load_model(model_path)
    
    # 2. 读取数据（不要转datetime）
    data = pd.read_csv(data_path)
    
    # 3. 筛选指定product的数据
    product_data = data[data['product_pid'] == product_pid].sort_values(by='transaction_date')
    
    if len(product_data) < 10:
        raise ValueError(f"Product {product_pid} 的数据量不足10条记录")
    
    # 4. 获取最后10天的数据作为特征
    last_10_days = product_data.tail(10)
    feature_cols = [col for col in data.columns if col not in ['product_pid']]
    X_single = last_10_days[feature_cols]
    
    # 5. 预测未来10天
    predictions = model.predict(X_single)
    
    # 6. 生成未来10个交易日的日期
    last_date_raw = last_10_days['transaction_date'].iloc[-1]
    last_date_str = str(last_date_raw)
    
    # 读取交易日信息
    time_info = pd.read_csv('data/time_info_final.csv')
    time_info['stat_date'] = time_info['stat_date'].astype(str)
    
    # 找到最后一个交易日在time_info中的位置
    last_date_idx = time_info[time_info['stat_date'] == last_date_str].index[0]
    
    # 获取未来10个交易日
    future_trading_dates = []
    current_idx = last_date_idx + 1
    count = 0
    
    while count < 10 and current_idx < len(time_info):
        if time_info.iloc[current_idx]['is_trade'] == 1:
            future_trading_dates.append(time_info.iloc[current_idx]['stat_date'])
            count += 1
        current_idx += 1
    
    # 格式化日期
    future_dates = [datetime.strptime(date, "%Y%m%d").strftime('%Y-%m-%d') for date in future_trading_dates]
    
    # 7. 整理预测结果
    future_predictions = []
    for i, date in enumerate(future_dates):
        day_prediction = {
            'date': date,
            'apply_amt_pred': float(predictions[i][0]),
            'redeem_amt_pred': float(predictions[i][1]),
            'net_in_amt_pred': float(predictions[i][2]),
            'net_flow_calculated': float(predictions[i][0] - predictions[i][1])
        }
        future_predictions.append(day_prediction)
    
    # 8. 返回结果
    result = {
        'product_pid': product_pid,
        'prediction_start_date': future_dates[0],
        'prediction_end_date': future_dates[-1],
        'future_predictions': future_predictions,
        'summary': {
            'avg_apply_amt': float(sum(p['apply_amt_pred'] for p in future_predictions) / 10),
            'avg_redeem_amt': float(sum(p['redeem_amt_pred'] for p in future_predictions) / 10),
            'avg_net_in_amt': float(sum(p['net_in_amt_pred'] for p in future_predictions) / 10),
            'total_net_flow': float(sum(p['net_flow_calculated'] for p in future_predictions))
        }
    }
    
    return result

def print_prediction_results(result):
    """
    打印预测结果
    """
    print(f"\n{'='*60}")
    print(f"Product {result['product_pid']} 未来10天预测结果")
    print(f"{'='*60}")
    print(f"预测时间范围: {result['prediction_start_date']} 到 {result['prediction_end_date']}")
    print(f"{'='*60}")
    
    print(f"\n{'日期':<12} {'申请金额':<12} {'赎回金额':<12} {'净流入':<12} {'计算净流入':<12}")
    print("-" * 60)
    
    for pred in result['future_predictions']:
        print(f"{pred['date']:<12} {pred['apply_amt_pred']:<12.4f} {pred['redeem_amt_pred']:<12.4f} "
              f"{pred['net_in_amt_pred']:<12.4f} {pred['net_flow_calculated']:<12.4f}")
    
    print("-" * 60)
    print(f"\n{'汇总统计':<20}")
    print(f"{'平均申请金额':<20}: {result['summary']['avg_apply_amt']:.4f}")
    print(f"{'平均赎回金额':<20}: {result['summary']['avg_redeem_amt']:.4f}")
    print(f"{'平均净流入':<20}: {result['summary']['avg_net_in_amt']:.4f}")
    print(f"{'10天总净流入':<20}: {result['summary']['total_net_flow']:.4f}")

# 使用示例
if __name__ == "__main__":
    # 示例1: 预测单个product未来10天
    # try:
    # 你需要替换为实际存在的product_pid
    product_id = "product1"  # 替换为你要预测的product_pid
    
    result = predict_future_10_days(product_id)
    print(result['future_predictions'])
    
    # 打印预测结果
    # print_prediction_results(result)
        
        
    # except Exception as e:
    #     print(f"预测失败: {str(e)}")
    
