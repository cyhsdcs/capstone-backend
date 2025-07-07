from typing import Union

from fastapi import FastAPI
from predict_by_CNN import predict_by_CNN
import pandas as pd
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
