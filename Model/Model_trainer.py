#encoding=utf-8
import time
import os

import numpy as np
import torch
from torch import nn
from script import metrics




def train(model, optimizer, train_dataloader, test_dataloader, device, epochs, data_mean, data_std, tdata_mean, tdata_std):
    model.train()
    #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1)
    s1_loss = nn.SmoothL1Loss()
    mean_mse_epoch = list()
    Time = 0
    
    bMAE = 100
    bWMAPE = 100
    bRMSE = 100

    for epoch in range(epochs):
        start_time = time.time()
        S1sum = 0
        num = 0
        tnum = 0
        MAEsum = 0
        
        #
        tMAEsum = 0
        tWMAPEsum = 0
        tRMSEsum = 0

        for data in train_dataloader:
            data['flow_x'], data['flow_y'], data['week_cyc'], data['day_cyc'] =  data['flow_x'].to(device), data[
                'flow_y'].to(device), data['week_cyc'].to(device), data['day_cyc'].to(device)            
            
            
            
            y_data = data['flow_y']
            predit = model(data)
            
            y_real = y_data.detach().cpu().numpy()
            y_pred = predit.detach().cpu().numpy()
            
            num = num + 1
            # 
            y_real = y_real * data_std + data_mean
            y_pred = y_pred * data_std + data_mean
            
            MAE = metrics.MAE(y_real, y_pred)
            RMSE = metrics.RMSE(y_real, y_pred)
            MAEsum = MAE + MAEsum
        
            s1 = s1_loss(predit, y_data)
            S1 = s1.detach().cpu().numpy()
            S1sum = S1sum + S1
            
            optimizer.zero_grad()
            s1.backward()
            optimizer.step()
        end_time = time.time()
        Time += (end_time - start_time)
        print("train epoch: {}, 平滑L1 Loss:{}, MAE Loss:{}, train time:{}".format(epoch, S1sum/num, MAEsum/num, end_time - start_time))
        

        for tdata in test_dataloader:
            tdata['graph'], tdata['flow_x'], tdata['flow_y'] = tdata['graph'].to(device), tdata['flow_x'].to(device), tdata[
                'flow_y'].to(device)
            y_tdata = tdata['flow_y']
            tpredict = model(tdata)


            y_test_real = y_tdata.detach().cpu().numpy()
            y_test_pred = tpredict.detach().cpu().numpy()
            
            #
            y_test_r = y_test_real * tdata_std + tdata_mean
            y_test_p = y_test_pred * tdata_std + tdata_mean

            tnum = tnum + 1
            
            tMAE = metrics.MAE(y_test_r, y_test_p)
            tMAEsum = tMAEsum + tMAE
            
            tWMAPE = metrics.MAPE(y_test_real, y_test_pred)
            tWMAPEsum = tWMAPEsum + tWMAPE

            tRMSE = metrics.RMSE(y_test_r, y_test_p)
            tRMSEsum = tRMSEsum + tRMSE
        
        if tMAEsum/tnum < bMAE:
            bMAE = tMAEsum/tnum
            bWMAPE = tWMAPEsum/tnum
            bRMSE = tRMSEsum/tnum

        print("test epoch:{}, MAE:{}, MAPE:{}, RMSE:{}".format(epoch,tMAEsum/tnum, tWMAPEsum/tnum, tRMSEsum/tnum))
    print("train time:{}".format(Time))
    print("*************************************************")
    

    print("test sets performence: ")
    print("MAE:{}, MAPE:{}, RMSE:{}".format(bMAE, bWMAPE, bRMSE))
    #if os.path.exists('./model_saved/GCTAN_829_12_08.pth'):
        #os.remove('./model_saved/GCTAN_829_12_08.pth')
    torch.save(model.state_dict(), './model_saved/GCTAN_829_12_08.pth')



