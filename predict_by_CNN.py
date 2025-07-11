import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import argparse
import sys
import datetime
from pathlib import Path
import yaml
import math
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

test_info = pd.read_csv('test_data.csv')
total_dataset = pd.read_csv('total_dataset.csv')
feature_list = ['product_pid',
                'uv_fundown', 'uv_stableown', 'uv_fundopt', 'uv_fundmarket', 'uv_termmarket',
                'is_trade', 'next_trade_date','last_trade_date','is_week_end', 'is_month_end','is_quarter_end','is_year_end','trade_day_rank','yield']
label_list = ['apply_amt', 'redeem_amt','net_in_amt']  # 预测的标签

# 使用十天数据预测后十天
window_config = {
    'feature_size': 10,
    'label_size': 10
}

### Net
class Net(nn.Module):
    def __init__(self, num_label):
        super(Net, self).__init__()
        self.num_label = num_label
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 5, 1, 1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 3)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Linear(64, 16)
        self.norm1 = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(16, self.num_label*10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = x.unsqueeze(1)
        return x

def get_args():
    parser = argparse.ArgumentParser('time series forecasting in the financial sector')

    parser.add_argument('--cpu', type=bool, default=True, help='wheather to use cpu , if not set, use gpu')

    parser.add_argument('--saved_dir', type=str, default='./model/CNN')

    parser.add_argument('--pid', type=str, default='1', help='product pid')

    args = parser.parse_args()

    if not args.cpu and not torch.cuda.is_available():
        logger.error('cuda is not available, try running on CPU by adding param --cpu')
        sys.exit()

    args.begin_time = datetime.datetime.now().strftime('%H%M%S')
    args.saved_dir = Path(args.saved_dir)
    logger.info(f'Save Dir : {str(args.saved_dir)}')    

    with open(args.saved_dir / 'opt.yaml','w') as f:
        yaml.dump(vars(args),f,sort_keys=False)

    return args


### 预测
def predict(pid):
    device = torch.device('cpu')
    model = Net(len(label_list)).to(device)
    checkpoint = torch.load('./model/CNN/model.pth')
    model.load_state_dict(checkpoint['modelstate'])

    total_data_group_product = total_dataset[feature_list].groupby('product_pid')

    model.eval()
    with torch.no_grad():
        pid_id = int(pid)
        # 检查产品是否存在
        if pid_id not in total_data_group_product.groups:
            logger.error(f'Product ID {pid_id} not found in dataset')
            return
            
        # 获取该产品的数据
        product_data = total_data_group_product.get_group(pid_id)
            

        feat_np = np.array(product_data.tail(window_config['feature_size']))
        feature = torch.from_numpy(feat_np).float().unsqueeze(0).unsqueeze(0).to(device)
        
        # 进行预测
        pred = model(feature)
        pred_th = torch.from_numpy(pred.cpu().numpy())
        B= pred_th.shape[0]
        pred_th = pred_th.squeeze().reshape(3,-1)
        apply_amt_pred = pred_th[0].squeeze().numpy().tolist()
        redeem_amt_pred = pred_th[1].squeeze().numpy().tolist()
        net_in_amt_pred = (0.5*pred_th[2].squeeze().numpy()+0.5*(pred_th[0].squeeze().numpy() - pred_th[1].squeeze().numpy())).tolist()
        time_list = ['20221110', '20221111', '20221114', '20221115', '20221116', '20221117', '20221118', '20221121', '20221122', '20221123' ]
        return time_list, apply_amt_pred, redeem_amt_pred, net_in_amt_pred


def predict_by_CNN(pid: str):
    return predict(pid)

if __name__ == '__main__':
    print(predict_by_CNN('1'))