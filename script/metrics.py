import math

import numpy as np

def mask_np(array, null_val):
    if np.isnan(null_val):
        return (~np.isnan(null_val)).astype('float32')
    else:
        return np.not_equal(array, null_val).astype('float32')

def MAE(y_true, y_pre):
    y_true = (y_true).copy().reshape((-1, 1))
    y_pre = (y_pre).copy().reshape((-1, 1))
    re = np.abs(y_true - y_pre).mean()
    return re


def RMSE(y_true, y_pre):
    y_true = (y_true).copy().reshape((-1, 1))
    y_pre = (y_pre).copy().reshape((-1, 1))
    re = math.sqrt(((y_true - y_pre) ** 2).mean())
    return re

def MAPE2(y_true, y_pred, null_val=0):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = mask_np(y_true, null_val)
        mask /= mask.mean()
        mape = np.abs((y_pred - y_true) / y_true)
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100

def MAPE(y_true, y_pred, null_val=0.0):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'),
                      y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100



def MAPE1(y_true, y_pre):
    y_true = (y_true).copy().reshape((-1, 1))
    y_pre = (y_pre).copy().reshape((-1, 1))
    #e = (y_true + y_pre) / 2 + 1e-2
    re = (np.abs(y_true - y_pre) / (np.abs(y_true))).mean()
    return re*100


def WMAPE(y_true, y_pre):
    y_true = y_true.copy().reshape((-1, 1)).squeeze()
    y_pre = y_pre.copy().reshape((-1, 1)).squeeze()

    wmape = np.sum(np.abs(y_pre - y_true)) / np.sum(y_true)

    return wmape*100
