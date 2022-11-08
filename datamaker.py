from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
from utils.timefeatures import time_features


class Dataset_BOOLV(Dataset):
    def __init__(self, data_path, flag, size=None, features=None, usescale=True ,timeenc=1):
        self.data_path = data_path
        self.near_w, self.season_w, self.trend_w, self.predict_w = size
        typemap = {'train': 0, 'valid': 1, 'test': 2}
        assert flag in typemap.keys()
        self.data_type = typemap[flag]
        self.features = features
        self.usescale = usescale
        self.timeenc= timeenc
        self.__read_data()

    def __read_data(self):
        df_raw = pd.read_csv(self.data_path, parse_dates=['时间'])
        df_data0 = df_raw[:365][self.features].reset_index(drop=True)[90:-15].reset_index(drop=True)
        df_data1 = df_raw[-365:][self.features].reset_index(drop=True)[90:-15].reset_index(drop=True)
        # print(df_data0.describe(), df_data1.describe())
        
        window = self.season_w // 2 + 30 + self.predict_w
        data_len = len(df_data1) - window + 1 # 215 - 39 + 1 = 177
        num_test = 28
        num_train = int((data_len - num_test) * 0.9) # 134
        num_valid = data_len - num_train - num_test # 15
        # num_test = 28
        # num_valid = 28
        # num_train = data_len - num_valid - num_test
        num_len = [num_train, num_valid, num_test]

        startpoint = self.season_w // 2 + 30 - 1
        Ls = [startpoint, startpoint + num_train, startpoint + num_train + num_valid]
        Rs = [startpoint + num_train, startpoint + num_train + num_valid, startpoint + num_train + num_valid + num_test]
        #valid data point: train:[34: 168] valid[168:183] test[183:211]
        #range of data:    train:[34-35+1= 0:168+4= 172], valid[168-35+1= 134: 183+4= 187], test=[183-35+1= 149:211+4= 215]
        L, R = Ls[self.data_type], Rs[self.data_type]

        if self.data_type == 0:
            plt.plot(df_data1.index[Ls[0]:Rs[0]], df_data1[Ls[0]:Rs[0]])
            plt.plot(df_data1.index[Ls[1]:Rs[1]], df_data1[Ls[1]:Rs[1]])
            plt.plot(df_data1.index[Ls[2]:], df_data1[Ls[2]:])
            plt.plot(df_data0.index, df_data0)
            plt.show()

        if self.usescale:
            self.scaler0 = MinMaxScaler()
            self.scaler0.fit(df_data0.values)
            data0 = self.scaler0.transform(df_data0.values)

            self.scaler1 = MinMaxScaler()
            train_data = df_data1[: Rs[0] + self.predict_w]
            self.scaler1.fit(train_data.values)
            data1 = self.scaler1.transform(df_data1.values)
        else:
            data0 = df_data0.values
            data1 = df_data1.values
        
        df_stamp = df_raw[['时间']][-365:][90:-15]
        df_stamp['date'] = pd.to_datetime(df_stamp['时间'])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            data_stamp = df_stamp.drop(['date', '时间'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq='d')
            data_stamp = data_stamp.transpose(1, 0)

        self.data0 = data0
        self.data1 = data1
        self.data_stamp = data_stamp
        self.range = (L, R)
        self.len = num_len[self.data_type]

    def __getitem__(self, index):
        s_point = index + self.range[0]

        near_seq = self.data1[s_point+1-self.near_w: s_point+1]
        season_seq = self.data1[s_point-30+1-self.season_w//2: s_point-30+1+self.season_w//2]
        trend_seq = self.data0[s_point+1-(self.trend_w-self.predict_w): s_point+1+self.predict_w]
        predict_seq = self.data1[s_point+1: s_point+1+self.predict_w]

        near_stamp = self.data_stamp[s_point+1-self.near_w: s_point+1]
        season_stamp = self.data_stamp[s_point-30+1-self.season_w//2: s_point-30+1+self.season_w//2]
        trend_stamp = self.data_stamp[s_point+1-(self.trend_w-self.predict_w): s_point+1+self.predict_w]

        x_seq = (near_seq, season_seq, trend_seq)
        x_stamp = (near_stamp, season_stamp, trend_stamp)
        x, y = (x_seq, x_stamp), predict_seq
        return x, y

    def __len__(self):
        return self.len

    def inverse_transform(self, data):
        return self.scaler1.inverse_transform(data)


def generate_dataloader(args, flag='train'):
    timeenc = 0 if args.embed != 'timeF' else 1
    if flag == 'test':
        shuffle_flag = False
        batch_size = 1
    else:
        shuffle_flag = True
        batch_size = args.batch_size
    near_seq_window, season_seq_window = args.near_seq_window, args.season_seq_window
    trend_seq_window, predict_seq_window = args.trend_seq_window, args.predict_seq_window
    data_set = Dataset_BOOLV(
        data_path=args.data_path,
        flag=flag,
        size=[near_seq_window, season_seq_window, trend_seq_window, predict_seq_window],
        features=args.catelist,
        timeenc=timeenc)

    data_loader = DataLoader(
        dataset=data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers)

    return data_loader
