from pathlib import Path
from typing import Union

import numpy as np

import torch.nn as nn
import torch
import torch.nn.functional as F

from models.tacotron import CBHG


class BatchNormConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel, activation=None):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel, stride=1, padding=kernel // 2, bias=False)
        self.bnorm = nn.BatchNorm1d(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.activation:
            x = self.activation(x)
        x = self.bnorm(x)
        return x


class DurationPredictorModel(nn.Module):

    def __init__(self, num_chars, embed_dims, bits, conv_dims=256, rnn_dims=64, dropout=0.5):
        super().__init__()
        self.encoder_embedding = nn.Embedding(num_chars, embed_dims)
        self.convs = torch.nn.ModuleList([
            BatchNormConv(embed_dims, conv_dims, 5, activation=torch.relu),
            BatchNormConv(conv_dims, conv_dims, 5, activation=torch.relu),
            BatchNormConv(conv_dims, conv_dims, 5, activation=torch.relu),
        ])
        self.encoder = nn.GRU(conv_dims, rnn_dims, batch_first=True, bidirectional=True)
        self.decoder = nn.GRU(2*rnn_dims + embed_dims, rnn_dims, batch_first=True, bidirectional=False)
        self.dropout = dropout
        self.register_buffer('step', torch.zeros(1, dtype=torch.long))
        self.n_classes = 2**bits
        self.decoder_embedding = nn.Embedding(self.n_classes, embed_dims)
        self.bits = bits
        self.lin = nn.Linear(rnn_dims, self.n_classes)
        self.rnn_dims = rnn_dims

    def forward(self, x, target, alpha=1.0):
        device = next(self.parameters()).device  # use same device as parameters
        bsize = x.size(0)
        h1 = torch.zeros(1, bsize, self.rnn_dims, device=device)
        t1 = torch.zeros(bsize, 1, device=device).long()
        target = torch.cat([t1, target[:, :-1]], dim=1)
        if self.training:
            self.step += 1
        x = self.encoder_embedding(x)
        x = x.transpose(1, 2)
        for conv in self.convs:
            x = conv(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(1, 2)
        x, _ = self.encoder(x)

        target = self.decoder_embedding(target)
        x = torch.cat([x, target], dim=-1)
        x, _ = self.decoder(x, h1)

        x = self.lin(x)
        #x = x.squeeze()
        return x / alpha

    def generate(self, x, alpha=1.0):
        self.eval()
        device = next(self.parameters()).device  # use same device as parameters
        x = torch.as_tensor(x, dtype=torch.long, device=device).unsqueeze(0)
        x = self.encoder_embedding(x)
        x = x.transpose(1, 2)
        for conv in self.convs:
            x = conv(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(1, 2)
        x, _ = self.encoder(x)
        bsize = x.size(0)
        h1 = torch.zeros(bsize, self.rnn_dims, device=device)
        decoder = self.get_gru_cell(self.decoder)
        seq_len = x.shape[1]
        output = []
        target = torch.zeros(bsize, device=device).long()
        for i in range(seq_len):
            target = self.decoder_embedding(target)
            x_t = x[:, i, :]
            x_t = torch.cat([x_t, target], dim=-1)
            h1 = decoder(x_t, h1)
            x_t = self.lin(h1)
            posterior = F.softmax(x_t, dim=1)
            sample = torch.distributions.Categorical(posterior).sample().float()
            output.append(sample)
            target = sample.long()
        output = torch.stack(output).transpose(0,1).long()
        print(output)
        return output / alpha

    def get_gru_cell(self, gru):
        gru_cell = nn.GRUCell(gru.input_size, gru.hidden_size)
        gru_cell.weight_hh.data = gru.weight_hh_l0.data
        gru_cell.weight_ih.data = gru.weight_ih_l0.data
        gru_cell.bias_hh.data = gru.bias_hh_l0.data
        gru_cell.bias_ih.data = gru.bias_ih_l0.data
        return gru_cell

    def get_step(self):
        return self.step.data.item()

    def load(self, path: Union[str, Path]):
        # Use device of model params as location for loaded state
        device = next(self.parameters()).device
        state_dict = torch.load(path, map_location=device)
        self.load_state_dict(state_dict, strict=False)

    def save(self, path: Union[str, Path]):
        # No optimizer argument because saving a model should not include data
        # only relevant in the training process - it should only be properties
        # of the model itself. Let caller take care of saving optimzier state.
        torch.save(self.state_dict(), path)

    def log(self, path, msg):
        with open(path, 'a') as f:
            print(msg, file=f)