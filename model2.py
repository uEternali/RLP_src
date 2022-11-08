import torch
from torch import  nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, embedding_size):
        super(TimeFeatureEmbedding, self).__init__()
        d_inp = 4
        self.embed = nn.Linear(d_inp, embedding_size, bias=False)

    def forward(self, x):
        return self.embed(x)

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, embedding_size):
        super(TokenEmbedding, self).__init__()
        padding = 1
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=embedding_size,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

class Series_Encoder(nn.Module):
    def __init__(self, feature_size, hidden_size):
        super().__init__()
        self.feature_size = feature_size
        self.embedding_size = 8
        self.hidden_size = hidden_size

        
        self.value_embedding = TokenEmbedding(c_in=feature_size, embedding_size=self.embedding_size)
        self.time_embedder = TimeFeatureEmbedding(embedding_size=self.embedding_size)

        self.lstm_embedder = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size, 
            batch_first=True)

    def forward(self, seq: torch.Tensor, stamp: torch.Tensor):
        seq_embedding = self.value_embedding(seq)
        time_embedding = self.time_embedder(stamp)
        data = seq_embedding + time_embedding
        enc_output, (enc_hidden, enc_cell) = self.lstm_embedder(data)
        
        return enc_output

class Attention(nn.Module):
    def __init__(self, s_dim, enc_hid_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hid_dim + s_dim, enc_hid_dim, bias=False)
        self.v = nn.Linear(enc_hid_dim, 1, bias = False)

        # self.mask = nn.Parameter(torch.tensor([0.0, 1, 2, 3, 4], device=device).view(1, -1))
        
    def forward(self, s:torch.Tensor, enc_output):
        
        src_len = enc_output.shape[1]
        
        s = s.unsqueeze(1).contiguous().repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((s, enc_output), dim = 2)))
        attention = self.v(energy).squeeze(2)

        # s = s.unsqueeze(1).contiguous()
        # attention = (torch.bmm(enc_output, s.transpose(1, 2)) / torch.sqrt(torch.tensor(s.shape[2], device=device))).squeeze(2)

        # s = s.unsqueeze(1).contiguous()
        # v = self.w(enc_output)
        # attention = torch.bmm(v, s.transpose(1 ,2)).squeeze(2)

        # score = F.softmax(attention, dim=1).unsqueeze(1)
        # attn_value = torch.bmm(score, enc_output)

        # score = F.softmax(self.mask, dim=1).unsqueeze(1)
        # attn_value = torch.bmm(score.repeat(enc_output.shape[0], 1, 1), enc_output)

        # s = s.unsqueeze(1).contiguous().repeat(1, src_len, 1)
        # attention = torch.tanh(self.attn(torch.cat((s, enc_output), dim = 2))).squeeze(2)

        score = F.softmax(attention, dim=1).unsqueeze(1)
        attn_value = torch.bmm(score, enc_output)
        
        return attn_value

class RLPnet(nn.Module):
    def __init__(self, feature_size, pred_w, attn_src_len, use_seaon, use_trend):
        super().__init__()
        self.use_season = use_seaon
        self.use_trend = use_trend

        self.pred_w = pred_w
        self.feature_size = feature_size
        self.attn_src_len = attn_src_len

        self.near_hidden_size = 64
        self.season_hidden_size = 64
        self.trend_hidden_size = 64

        hidden_size = self.near_hidden_size
        self.near_encoder = Series_Encoder(self.feature_size, self.near_hidden_size)
        self.season_encoder = Series_Encoder(self.feature_size, self.season_hidden_size)
        self.trend_encoder = Series_Encoder(self.feature_size, self.trend_hidden_size)

        self.season_attention = Attention(hidden_size, hidden_size)
        self.year_attention = Attention(hidden_size, hidden_size)

        # self.MLP_layer = nn.Linear(embedding_size * 3, 1)
        self.MLP_layer = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, self.pred_w)
        )


    def forward(self, x: torch.Tensor, stamp: torch.Tensor):
        near_seq, season_seq, trend_seq = x
        near_stamp, season_stamp, trend_stamp = stamp

        near_output = self.near_encoder(near_seq, near_stamp)

        season_output = self.season_encoder(season_seq, season_stamp)

        year_output = self.trend_encoder(trend_seq, trend_stamp)
        
        # near_hidden = near_output[:, -1, :]
        # season_hidden = season_output[:, -1, :]
        # year_hidden = year_output[:, -1, :]
        # hidden_vec = near_hidden + season_hidden + year_hidden

        near_hidden = near_output[:, -1, :]
        season_hidden_attn = self.season_attention(near_hidden, season_output[:, -self.attn_src_len:, :])
        year_hidden_attn = self.year_attention(near_hidden, year_output[:, -self.attn_src_len:, :])
        hidden_vec = near_hidden + season_hidden_attn.squeeze(1) + year_hidden_attn.squeeze(1)

        output = self.MLP_layer(hidden_vec)

        return torch.tanh(output).view(-1, self.pred_w, 1)
