import math
from argparse import Namespace

import torch
from torch import nn, Tensor


class MultiHeadedAttention(nn.Module):
    def __init__(self, heads: int, d_model: int) -> None:
        super(MultiHeadedAttention, self).__init__()
        assert d_model % heads == 0

        self.d_k = d_model // heads
        self.heads = heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor) -> Tensor:
        n_batches = x.shape[0]

        Q = self.W_q(x).reshape(n_batches, -1, self.heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).reshape(n_batches, -1, self.heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).reshape(n_batches, -1, self.heads, self.d_k).transpose(1, 2)

        x = MultiHeadedAttention.attention(Q, K, V)
        x = x.transpose(1, 2).contiguous().reshape(n_batches, -1, self.heads * self.d_k)

        return self.W_o(x)

    @staticmethod
    def attention(Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
        assert Q.shape[-1] == K.shape[-1]

        d_k = Q.shape[-1]
        K_T = K.transpose(-2, -1)
        scores = torch.matmul(Q, K_T) / math.sqrt(d_k)
        p_attn = scores.softmax(dim=-1)

        return torch.matmul(p_attn, V)


class AssociatedAttention(nn.Module):
    def __init__(self, head_dims: list, d_model: int) -> None:
        super(AssociatedAttention, self).__init__()

        self.head_dims = head_dims
        for dims in head_dims:
            for dim in dims:
                assert 0 <= dim < d_model

        dim_size = 0
        self.W_qs = nn.ModuleList()
        self.W_ks = nn.ModuleList()
        self.W_vs = nn.ModuleList()
        for dims in head_dims:
            dim_size += len(dims)
            self.W_qs.append(nn.Linear(len(dims), len(dims)))
            self.W_ks.append(nn.Linear(len(dims), len(dims)))
            self.W_vs.append(nn.Linear(len(dims), len(dims)))

        self.W_o = nn.Linear(dim_size, d_model)

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndimension() == 3

        n_batches = x.shape[0]

        Qs = list()
        Ks = list()
        Vs = list()
        for h, dims in enumerate(self.head_dims):
            a_dims = x.index_select(dim=2, index=torch.tensor(dims, device=x.device))
            Qs.append(self.W_qs[h](a_dims).reshape(n_batches, -1, 1, len(dims)).transpose(1, 2))
            Ks.append(self.W_ks[h](a_dims).reshape(n_batches, -1, 1, len(dims)).transpose(1, 2))
            Vs.append(self.W_vs[h](a_dims).reshape(n_batches, -1, 1, len(dims)).transpose(1, 2))

        outs = list()
        for h, dims in enumerate(self.head_dims):
            a_out = AssociatedAttention.attention(Qs[h], Ks[h], Vs[h])
            a_out = a_out.transpose(1, 2).contiguous().reshape(n_batches, -1, 1 * len(dims))
            outs.append(a_out)

        outs = torch.cat(outs, dim=2)
        outs = self.W_o(outs)

        return outs

    @staticmethod
    def attention(Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
        assert Q.shape[-1] == K.shape[-1]

        d_k = Q.shape[-1]
        K_T = K.transpose(-2, -1)
        scores = torch.matmul(Q, K_T) / math.sqrt(d_k)
        p_attn = scores.softmax(dim=-1)
        return torch.matmul(p_attn, V)


class ALEncoder(nn.Module):
    def __init__(self, d_model: int, head_dims: list, n_hidden: int, n_layers: int, dropout: float) -> None:
        super(ALEncoder, self).__init__()

        self.attn = AssociatedAttention(head_dims, d_model)
        # self.attn = MultiHeadedAttention(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=n_hidden, num_layers=n_layers, bidirectional=True, batch_first=True)

    def forward(self, x: Tensor) -> Tensor:
        out = self.attn(x)
        out = self.drop(out)
        out = self.norm(x + out)
        out, state = self.lstm(out)

        return out[:, -1]


class ConvBlock(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, kernel_size: int, stride: int, padding: int) -> None:
        super(ConvBlock, self).__init__()

        self.base = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.base(x)


class ConvEncoder(nn.Module):
    def __init__(self, d_model: int, channels: list, kernel_sizes: list, strides: list, paddings: list) -> None:
        super(ConvEncoder, self).__init__()
        assert len(channels) == len(kernel_sizes) == len(strides) == len(paddings)

        self.blocks = nn.Sequential()
        for channel, kernel_size, stride, padding in zip(channels, kernel_sizes, strides, paddings):
            self.blocks.append(ConvBlock(d_model, channel, kernel_size, stride, padding))

        # self.blocks.append(nn.Flatten())
        # self.blocks.append(nn.Linear(channels[-1], d_model))

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 1)

        out = [block(x) for block in self.blocks]
        out = torch.cat(out, dim=-1)
        out = out.view(x.shape[0], -1)

        return out


class Decoder(nn.Module):
    def __init__(self, dim_input: int, dim_forward: int, dim_output: int, dropout: float) -> None:
        super(Decoder, self).__init__()

        if dim_input == -1:
            self.linear1 = nn.LazyLinear(dim_forward)
        else:
            self.linear1 = nn.Linear(dim_input, dim_forward)

        self.drop = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_forward, dim_output)
        self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        out = self.linear1(x)
        out = self.activation(out)
        out = self.drop(out)
        out = self.linear2(out)

        return out


class EmotionPredictModel(nn.Module):
    def __init__(self, d_model: int, dim_output: int, args: Namespace) -> None:
        super(EmotionPredictModel, self).__init__()

        self.al_encoder = ALEncoder(d_model, args.head_dims, args.n_hidden, args.n_layers, args.dropout1)
        self.cn_encoder = ConvEncoder(d_model, args.channels, args.kernel_sizes, args.strides, args.paddings)
        self.decoder = Decoder(-1, args.dim_forward, dim_output, args.dropout2)

    def forward(self, x: Tensor) -> Tensor:
        out_1 = self.al_encoder(x)
        out_2 = self.cn_encoder(x)

        out = torch.cat([out_1, out_2], dim=-1)
        out = self.decoder(out)

        return out
