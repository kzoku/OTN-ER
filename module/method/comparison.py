import math
from argparse import Namespace

import numpy as np
import torch
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sktime.transformations.panel.rocket import Rocket
from torch import nn, Tensor
from tslearn.clustering import TimeSeriesKMeans, KShape
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.neural_network import TimeSeriesMLPClassifier

from module.config.config import Constant


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, data_len: int) -> None:
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(data_len, d_model)

        position = torch.arange(0, data_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model) * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term[0::2])
        pe[:, 1::2] = torch.cos(position * div_term[1::2])
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)

        return self.dropout(x)


class TimeSeriesModel(nn.Module):
    def __init__(self, num_inputs: int, data_len: int, num_outputs: int, args: Namespace) -> None:
        super(TimeSeriesModel, self).__init__()
        assert args.clf_type in Constant.TS_METHODS

        self.positional_encoding = PositionalEncoding(d_model=num_inputs, dropout=0, data_len=data_len)

        self.clf_type = args.clf_type
        if args.clf_type == Constant.RNN:
            self.encoder = nn.RNN(input_size=num_inputs, hidden_size=args.n_hidden, num_layers=args.n_layers, bidirectional=True, batch_first=True)
        elif args.clf_type == Constant.LSTM:
            self.encoder = nn.LSTM(input_size=num_inputs, hidden_size=args.n_hidden, num_layers=args.n_layers, bidirectional=True, batch_first=True)
        elif args.clf_type == Constant.GRU:
            self.encoder = nn.GRU(input_size=num_inputs, hidden_size=args.n_hidden, num_layers=args.n_layers, bidirectional=True, batch_first=True)
        elif args.clf_type == Constant.Transformer:
            self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=num_inputs, nhead=args.n_heads, batch_first=True), num_layers=args.n_layers)

        self.decoder = nn.Sequential()
        if args.clf_type in [Constant.RNN, Constant.GRU, Constant.LSTM]:
            if self.encoder.bidirectional:
                self.decoder.add_module('Layer1', nn.Linear(args.n_hidden * 2, args.n_hidden))
            else:
                self.decoder.add_module('Layer1', nn.Linear(args.n_hidden * 1, args.n_hidden))

            self.decoder.add_module('ReLU2', nn.ReLU())
            self.decoder.add_module('Layer3', nn.Linear(args.n_hidden, num_outputs))
        elif args.clf_type == Constant.Transformer:
            self.decoder.add_module('Layer1', nn.Linear(data_len * num_inputs, args.n_hidden))
            self.decoder.add_module('ReLU2', nn.ReLU())
            self.decoder.add_module('Layer3', nn.Linear(args.n_hidden, num_outputs))

    def forward(self, inputs: Tensor) -> Tensor:
        if self.clf_type in [Constant.RNN, Constant.GRU, Constant.LSTM]:
            self.encoder.flatten_parameters()

            if self.encoder.batch_first:
                outputs, state = self.encoder(inputs)

                encoding = outputs[:, -1]
            else:
                inputs = inputs.permute(1, 0, 2)

                outputs, state = self.encoder(inputs)

                encoding = outputs[-1]
        else:
            encoding = self.positional_encoding(inputs)
            encoding = self.encoder(encoding)

            encoding = encoding.view(encoding.size(0), -1)

        outs = self.decoder(encoding)
        return outs


class BaselineModel:
    def __init__(self, clf_type: str, n_class: int) -> None:
        assert clf_type in Constant.BL_METHODS

        self.clf_type = clf_type
        self.n_class = n_class
        self.clf = None

    def __call__(self, *args, **kwargs):
        train_x, train_y, test_x = args

        if self.clf_type == Constant.KNN:
            self.clf = KNeighborsTimeSeriesClassifier(n_neighbors=self.n_class, metric='dtw')
        elif self.clf_type == Constant.KMeans:
            self.clf = TimeSeriesKMeans(n_clusters=self.n_class, metric='dtw')
        elif self.clf_type == Constant.KShape:
            self.clf = KShape(n_clusters=self.n_class)
        elif self.clf_type == Constant.MLP:
            self.clf = TimeSeriesMLPClassifier(hidden_layer_sizes=(512, 512), random_state=0)
        elif self.clf_type == Constant.ROCKET:
            self.clf = make_pipeline(Rocket(), RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)))

        self.clf.fit(train_x, train_y)
        return self.clf.predict(test_x)


class SOTAModel:
    def __init__(self, clf_type: str, n_class: int) -> None:
        super(SOTAModel, self).__init__()
        assert clf_type in Constant.ST_METHODS

        self.clf_type = clf_type
        self.n_class = n_class
        self.clf = None

    def __call__(self, *args, **kwargs):
        train_x, train_y, test_x = args

        self.clf = KShape(n_clusters=self.n_class)
        self.clf.fit(train_x, train_y)

        return self.clf.predict(test_x)
