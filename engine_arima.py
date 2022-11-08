from statsmodels.tsa.arima.model import ARIMA
from utils.loggerinfo import get_logger
import copy
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np


class Engine_ARIMA():
    def __init__(self, args):
        self.args = args
        self.logger = get_logger('ARIMA.txt', 'arima')

    def get_data(self, cate, flag):
        # df_raw = pd.read_csv(self.args.data_path)[365 + 150:].reset_index(drop=True)
        # cate_data = df_raw[cate].values
        # if flag == 'train': return cate_data[:183]
        # else: return cate_data[183:]
        df_raw = pd.read_csv(self.args.data_path).reset_index(drop=True)
        cate_data = df_raw[cate].values
        if flag == 'train': return cate_data[ :-(15 + self.args.predict_seq_window)]
        else: return cate_data[-(15 + self.args.predict_seq_window):]

    def train(self):
        catelist = self.args.catelist
        mse_list, mae_list = [], []
        for cate in catelist:
            traindata = self.get_data(cate,'train')
            testdata = self.get_data(cate, 'test')
            scaler = StandardScaler()
            scaler.fit(traindata.reshape(-1, 1))
            traindata = scaler.transform(traindata.reshape(-1, 1)).reshape(-1).tolist()
            testdata = scaler.transform(testdata.reshape(-1, 1)).reshape(-1).tolist()

            prediction, groundtruth = [], []
            history = copy.deepcopy(traindata)

            predict_w = self.args.predict_seq_window
            for t in range(len(testdata) - predict_w):
                yhat, last = [], testdata[t]
                for pt in range(predict_w): 
                    nhistory = copy.deepcopy(history)
                    nhistory.append(last)
                    model = ARIMA(nhistory, order=(2, 1, 2))
                    model_fit = model.fit()
                    yhat.append(model_fit.forecast()[0])
                    last = model_fit.forecast()[0]
                prediction.append(yhat)
                groundtruth.append(testdata[t+1:t+1+predict_w])
                history.append(testdata[t])
            
            rmse_error = (np.sqrt(mean_squared_error(np.array(prediction), np.array(groundtruth)) * (scaler.scale_**2))).item()
            mae_error = (mean_absolute_error(np.array(prediction), np.array(groundtruth)) * (scaler.scale_)).item()
            self.logger.info(cate + 'RMSE: \t\t{:.6f}'.format(rmse_error))
            self.logger.info(cate + 'MEA: \t\t{:.6f}'.format(mae_error))
            mse_list.append(rmse_error**2)
            mae_list.append(mae_error)

        all_rmse_error = np.sqrt(np.mean(mse_list))
        all_mae_error = np.mean(mae_list)
        self.logger.info('ALL RMSE: \t\t{:.4f}'.format(all_rmse_error))
        self.logger.info('ALL MEA: \t\t{:.4f}'.format(all_mae_error))   