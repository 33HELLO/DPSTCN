# encoding=utf-8
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from script import metrics


def test(model, dataloader, data_max, data_min, pd_pth, device):
    model.eval()

    model.load_state_dict(torch.load(pd_pth))
    y_real = np.array([])
    y_pred = np.array([])
    num = 0
    MAEsum = 0
    RMSEsum = 0
    WMAPEsum = 0

    with torch.no_grad():
        for data in dataloader:
            data['graph'], data['flow_x'], data['flow_y'] = data['graph'].to(device), data['flow_x'].to(device), data[
                'flow_y'].to(device)
            y_data = data['flow_y']
            predict = model(data)

        
            y_real = y_data.detach().cpu().numpy()
            y_pred = predict.detach().cpu().numpy()

            y_real = y_real * (data_max - data_min) + data_min
            y_pred = y_pred * (data_max - data_min) + data_min
            
            num = num + 1

            MAE = metrics.MAE(y_real, y_pred)
            MAEsum = MAEsum + MAE

            RMSE = metrics.RMSE(y_real, y_pred)
            RMSEsum = RMSEsum + RMSE
            
            WMAPE = metrics.WMAPE(y_real, y_pred)
            WMAPEsum = WMAPEsum + WMAPE

        print("MAE:{}, WMAPE:{}%, RMSE:{}".format(MAEsum/num, WMAPEsum/num, RMSEsum/num))

        x = [ i for i in range(len(y_real[:, 0, :].flatten()))]
        plt.figure()
        plt.plot(x, y_real[:, 0, :].flatten(), label='real')
        plt.plot(x, y_pred[:, 0, :].flatten(), label='predict')
        plt.legend()
        plt.savefig('pred_real.png')
