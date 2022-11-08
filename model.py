from pyexpat import model
from turtle import st
import torch
from torch import dropout, nn
import math


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, embedding_size, embed_type='fixed'):
        super(TemporalEmbedding, self).__init__()

        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        self.weekday_embed = Embed(weekday_size, embedding_size)
        self.day_embed = Embed(day_size, embedding_size)
        self.month_embed = Embed(month_size, embedding_size)

    def forward(self, x):
        x = x.long()

        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return weekday_x + day_x + month_x

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, embedding_size, embed_type='timeF', freq='d'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, embedding_size, bias=False)

    def forward(self, x):
        return self.embed(x)

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, embedding_size):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=embedding_size,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

class Series_Embedding(nn.Module):
    def __init__(self, feature_size, embedding_size, embed_type='fixed'):
        super().__init__()
        self.feature_size = feature_size
        self.mid_embedding_size = 8
        self.embedding_size = embedding_size

        
        self.value_embedding = TokenEmbedding(c_in=feature_size, embedding_size=self.mid_embedding_size)
        if embed_type != 'timeF':
            self.time_embedder = TemporalEmbedding(embedding_size=self.mid_embedding_size, embed_type=embed_type) 
        else: self.time_embedder = TimeFeatureEmbedding(embedding_size=self.mid_embedding_size, embed_type=embed_type)

        self.lstm_embedder = nn.LSTM(input_size=self.mid_embedding_size, hidden_size=self.embedding_size, 
            batch_first=True)

        # for names in self.lstm_embedder._all_weights:
        #     for name in filter(lambda n: "bias" in n, names):
        #         bias = getattr(self.lstm_embedder, name)
        #         n = bias.size(0)
        #         start, end = n // 4, n // 2
        #         bias.data[start:end].fill_(1.)

    def forward(self, seq: torch.Tensor, stamp: torch.Tensor):
        seq_embedding = self.value_embedding(seq)
        time_embedding = self.time_embedder(stamp)
        data = seq_embedding + time_embedding
        _, (data_embedding, _) = self.lstm_embedder(data)
        data_embedding = data_embedding
        
        return data_embedding


class RLPnet(nn.Module):
    def __init__(self, feature_size, embed_type, use_seaon, use_trend):
        super().__init__()
        self.use_season = use_seaon
        self.use_trend = use_trend

        self.feature_size = feature_size
        self.embed_type = embed_type

        self.near_embedding_size = 64
        self.season_embedding_size = 64
        self.trend_embedding_size = 64

        embedding_size = self.near_embedding_size
        self.near_embedder = Series_Embedding(
            self.feature_size, self.near_embedding_size, self.embed_type)
        if self.use_season:
            embedding_size += self.season_embedding_size
            self.season_embedder = Series_Embedding(
                self.feature_size, self.season_embedding_size, self.embed_type)
        if self.use_trend:
            embedding_size += self.trend_embedding_size
            self.trend_embedder = Series_Embedding(
                self.feature_size, self.trend_embedding_size, self.embed_type)

        embedding_size = 64
        # self.MLP_layer = nn.Linear(embedding_size * 3, 1)
        self.MLP_layer = nn.Sequential(
            nn.Linear(embedding_size*1, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1)
        )


    def forward(self, x: torch.Tensor, stamp: torch.Tensor):
        near_seq, season_seq, trend_seq = x
        near_stamp, season_stamp, trend_stamp = stamp

        near_embedding = self.near_embedder(near_seq, near_stamp)
        embedding_vec = near_embedding

        if self.use_season:
            seaon_embedding = self.season_embedder(season_seq, season_stamp)
            embedding_vec = embedding_vec + seaon_embedding

        if self.use_trend:
            trend_embedding = self.trend_embedder(trend_seq, trend_stamp)
            embedding_vec = embedding_vec + trend_embedding
            
        embedding_vec = embedding_vec.permute(1, 2, 0).contiguous().view(embedding_vec.shape[1], -1)
        output = self.MLP_layer(embedding_vec)

        return torch.tanh(output).view(-1, 1, 1)
