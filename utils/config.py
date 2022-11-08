argsmap_testargs = {
    'argsname': 'testargs',
    'data_path': './data/cate_data.csv',
    'catelist': ['空调'],
    #['冰箱', '空调', '洗衣机', '电视'],

    'near_seq_window': 20,
    'season_seq_window': 10,
    'trend_seq_window': 10,
    'predict_seq_window':1,
    'attn_src_len': 3,
    
    'train_epochs': 300,
    'batch_size': 16,
    'learning_rate': 0.001,
    'weight_decay': 0,
    'patience': 20,
    'num_workers': 1,

    'embed': 'timeF'
}


class ArgsConfig:
    def __init__(self, argsmap):
        for argname, argvalue in argsmap.items():
            setattr(self, argname, argvalue)

    @classmethod
    def get_args(cls, argsmap=argsmap_testargs):
        return cls(argsmap)


if __name__ == '__main__':
    a = ArgsConfig.get_args()
    print(a.batch_size)
