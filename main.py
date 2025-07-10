from typing import Union

from fastapi import FastAPI
from predict_by_CNN import predict_by_CNN
import pandas as pd
from predict_by_XGB import predict_future_10_days
from transformer.predict_by_Transformer import predict_by_Transformer
app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/product")
def product():
    test_info = pd.read_csv('./test_data.csv')
    product_pids = test_info['product_pid'].unique().tolist()
    return {"product": product_pids}

@app.get("/cnn")
def cnn(pid: str = '1'):
    apply_amt_pred, redeem_amt_pred, net_in_amt_pred = predict_by_CNN(pid)
    return {"apply_amt_pred": apply_amt_pred, "redeem_amt_pred": redeem_amt_pred, "net_in_amt_pred": net_in_amt_pred}

@app.get("/xgboost")
def xgboost(pid: str = '1'):
    result = predict_future_10_days('product'+pid)
    return result['future_predictions']

@app.get("/transformer")
def transformer(pid: str = '1'):
    result = predict_by_Transformer(pid=pid, iflstm=False)
    return result

@app.get("/lstm")
def lstm(pid: str = '1'):
    result = predict_by_Transformer(pid=pid, iflstm=True)
    return result