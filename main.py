from typing import Union

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from predict_by_CNN import predict_by_CNN
import pandas as pd

from predict_by_XGB import predict_future_10_days
from transformer.predict_by_Transformer import predict_by_Transformer
app = FastAPI()

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://172.17.3.241:3000", "http://47.242.127.80:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/product")
def product():
    test_info = pd.read_csv('./test_data.csv')
    product_pids = sorted(test_info['product_pid'].unique().tolist(), key=int)
    return {"product": product_pids}

@app.get("/cnn")
def cnn(pid: str = '1'):
    time_list, apply_amt_pred, redeem_amt_pred, net_in_amt_pred = predict_by_CNN(pid)
    return {"time_list": time_list, "apply_amt_pred": apply_amt_pred, "redeem_amt_pred": redeem_amt_pred, "net_in_amt_pred": net_in_amt_pred}

@app.get("/xgboost")
def xgboost(pid: str = '1'):
    result = predict_future_10_days('product'+pid)
    return result['response']

@app.get("/transformer")
def transformer(pid: str = '1'):
    result = predict_by_Transformer(pid=pid, iflstm=False)
    return result

@app.get("/lstm")
def lstm(pid: str = '1'):
    result = predict_by_Transformer(pid=pid, iflstm=True)
    return result
