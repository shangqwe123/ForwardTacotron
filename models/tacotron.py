import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Union

from torch.nn import Embedding


class HighwayNetwork(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.W1 = nn.Linear(size, size)
        self.W2 = nn.Linear(size, size)
        self.W1.bias.data.fill_(0.)

    def forward(self, x):
        x1 = self.W1(x)
        x2 = self.W2(x)
        g = torch.sigmoid(x2)
        y = g * F.relu(x1) + (1. - g) * x
        return y


class Encoder(nn.Module):
    def __init__(self, embed_dims, num_chars, cbhg_channels, K, num_highways, dropout):
        super().__init__()
        self.embedding = nn.Embedding(num_chars, embed_dims)
        self.pre_net = PreNet(embed_dims)
        self.cbhg = CBHG(K=K, in_channels=cbhg_channels, channels=cbhg_channels,
                         proj_channels=[cbhg_channels, cbhg_channels],
                         num_highways=num_highways)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pre_net(x)
        x.transpose_(1, 2)
        x = self.cbhg(x)
        return x


class BatchNormConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, relu=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel, stride=1, padding=kernel // 2, bias=False)
        self.bnorm = nn.BatchNorm1d(out_channels)
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x) if self.relu is True else x
        return self.bnorm(x)


class CBHG(nn.Module):
    def __init__(self, K, in_channels, channels, proj_channels, num_highways):
        super().__init__()

        # List of all rnns to call `flatten_parameters()` on
        self._to_flatten = []

        self.bank_kernels = [i for i in range(1, K + 1)]
        self.conv1d_bank = nn.ModuleList()
        for k in self.bank_kernels:
            conv = BatchNormConv(in_channels, channels, k)
            self.conv1d_bank.append(conv)

        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

        self.conv_project1 = BatchNormConv(len(self.bank_kernels) * channels, proj_channels[0], 3)
        self.conv_project2 = BatchNormConv(proj_channels[0], proj_channels[1], 3, relu=False)

        # Fix the highway input if necessary
        if proj_channels[-1] != channels:
            self.highway_mismatch = True
            self.pre_highway = nn.Linear(proj_channels[-1], channels, bias=False)
        else:
            self.highway_mismatch = False

        self.highways = nn.ModuleList()
        for i in range(num_highways):
            hn = HighwayNetwork(channels)
            self.highways.append(hn)

        self.rnn = nn.GRU(channels, channels, batch_first=True, bidirectional=True)
        self._to_flatten.append(self.rnn)

        # Avoid fragmentation of RNN parameters and associated warning
        self._flatten_parameters()

    def forward(self, x):
        # Although we `_flatten_parameters()` on init, when using DataParallel
        # the model gets replicated, making it no longer guaranteed that the
        # weights are contiguous in GPU memory. Hence, we must call it again
        self._flatten_parameters()

        # Save these for later
        residual = x
        seq_len = x.size(-1)
        conv_bank = []

        # Convolution Bank
        for conv in self.conv1d_bank:
            c = conv(x) # Convolution
            conv_bank.append(c[:, :, :seq_len])

        # Stack along the channel axis
        conv_bank = torch.cat(conv_bank, dim=1)

        # dump the last padding to fit residual
        x = self.maxpool(conv_bank)[:, :, :seq_len]

        # Conv1d projections
        x = self.conv_project1(x)
        x = self.conv_project2(x)

        # Residual Connect
        x = x + residual

        # Through the highways
        x = x.transpose(1, 2)
        if self.highway_mismatch is True:
            x = self.pre_highway(x)
        for h in self.highways: x = h(x)

        # And then the RNN
        x, _ = self.rnn(x)
        return x

    def _flatten_parameters(self):
        """Calls `flatten_parameters` on all the rnns used by the WaveRNN. Used
        to improve efficiency and avoid PyTorch yelling at us."""
        [m.flatten_parameters() for m in self._to_flatten]

class PreNet(nn.Module):
    def __init__(self, in_dims, fc1_dims=256, fc2_dims=128, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.p = dropout

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, self.p, training=self.training)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, self.p, training=self.training)
        return x


class Attention(nn.Module):
    def __init__(self, attn_dims):
        super().__init__()
        self.W = nn.Linear(attn_dims, attn_dims, bias=False)
        self.v = nn.Linear(attn_dims, 1, bias=False)

    def forward(self, encoder_seq_proj, query, t):

        # print(encoder_seq_proj.shape)
        # Transform the query vector
        query_proj = self.W(query).unsqueeze(1)

        # Compute the scores
        u = self.v(torch.tanh(encoder_seq_proj + query_proj))
        scores = F.softmax(u, dim=1)

        return scores.transpose(1, 2)


class LSA(nn.Module):
    def __init__(self, attn_dim, kernel_size=31, filters=32, smooth_factor=10.):
        super().__init__()
        print(f'LSA using smooth_factor of {smooth_factor}')
        self.conv = nn.Conv1d(2, filters, padding=(kernel_size - 1) // 2, kernel_size=kernel_size, bias=False)
        self.L = nn.Linear(filters, attn_dim, bias=True)
        self.W = nn.Linear(attn_dim, attn_dim, bias=True)
        self.v = nn.Linear(attn_dim, 1, bias=False)
        self.cumulative = None
        self.attention = None
        self.t_max = None
        self.smooth_factor = smooth_factor

    def init_attention(self, encoder_seq_proj):
        device = next(self.parameters()).device  # use same device as parameters
        b, t, c = encoder_seq_proj.size()
        self.cumulative = torch.zeros(b, t, device=device)
        self.attention = torch.zeros(b, t, device=device)
        self.t_max = torch.zeros(b, device=device)

    def forward(self, encoder_seq_proj, query, t):
        device = next(self.parameters()).device  # use same device as parameters

        if t == 0:
            self.init_attention(encoder_seq_proj)

        processed_query = self.W(query).unsqueeze(1)

        location = torch.cat([self.cumulative.unsqueeze(1), self.attention.unsqueeze(1)], dim=1)
        processed_loc = self.L(self.conv(location).transpose(1, 2))

        u = self.v(torch.tanh(processed_query + encoder_seq_proj + processed_loc))
        u = u.squeeze(-1)

        b, tlen, _ = encoder_seq_proj.size()
        t_range = torch.arange(0, tlen-1, device=device)
        diff = (t_range[None, :] - self.t_max[:, None]) ** 2
        u = u - diff / self.smooth_factor

        scores = F.softmax(u, dim=1)
        values, indices = scores.max(1)
        self.t_max = indices

        self.attention = scores
        self.cumulative += self.attention


        #print(f'{t} {indices}')

        return scores.unsqueeze(-1).transpose(1, 2)


class Decoder(nn.Module):
    # Class variable because its value doesn't change between classes
    # yet ought to be scoped by class because its a property of a Decoder
    max_r = 20
    def __init__(self, n_mels, decoder_dims, lstm_dims, smooth_factor=10.):
        super().__init__()
        self.register_buffer('r', torch.tensor(1, dtype=torch.int))
        self.n_mels = n_mels
        self.prenet = PreNet(n_mels)
        self.attn_net = LSA(decoder_dims, smooth_factor=smooth_factor)
        self.attn_rnn = nn.GRUCell(decoder_dims + 128, decoder_dims)
        self.rnn_input = nn.Linear(2 * decoder_dims, lstm_dims)
        self.res_rnn1 = nn.LSTMCell(lstm_dims, lstm_dims)
        self.res_rnn2 = nn.LSTMCell(lstm_dims, lstm_dims)
        self.mel_proj = nn.Linear(lstm_dims, n_mels * self.max_r, bias=False)


    def zoneout(self, prev, current, p=0.1):
        device = next(self.parameters()).device  # Use same device as parameters
        mask = torch.zeros(prev.size(), device=device).bernoulli_(p)
        return prev * mask + current * (1 - mask)

    def forward(self, encoder_seq, encoder_seq_proj, prenet_in,
                hidden_states, cell_states, context_vec, t):

        # Need this for reshaping mels
        batch_size = encoder_seq.size(0)

        # Unpack the hidden and cell states
        attn_hidden, rnn1_hidden, rnn2_hidden = hidden_states
        rnn1_cell, rnn2_cell = cell_states

        # PreNet for the Attention RNN
        prenet_out = self.prenet(prenet_in)

        # Compute the Attention RNN hidden state
        attn_rnn_in = torch.cat([context_vec, prenet_out], dim=-1)

        attn_hidden = self.attn_rnn(attn_rnn_in.squeeze(1), attn_hidden)

        # Compute the attention scores
        scores = self.attn_net(encoder_seq_proj, attn_hidden, t)

        # Dot product to create the context vector
        context_vec = scores @ encoder_seq
        context_vec = context_vec.squeeze(1)

        # Concat Attention RNN output w. Context Vector & project
        x = torch.cat([context_vec, attn_hidden], dim=1)
        x = self.rnn_input(x)

        # Compute first Residual RNN
        rnn1_hidden_next, rnn1_cell = self.res_rnn1(x, (rnn1_hidden, rnn1_cell))
        if self.training:
            rnn1_hidden = self.zoneout(rnn1_hidden, rnn1_hidden_next)
        else:
            rnn1_hidden = rnn1_hidden_next
        x = x + rnn1_hidden

        # Compute second Residual RNN
        rnn2_hidden_next, rnn2_cell = self.res_rnn2(x, (rnn2_hidden, rnn2_cell))
        if self.training:
            rnn2_hidden = self.zoneout(rnn2_hidden, rnn2_hidden_next)
        else:
            rnn2_hidden = rnn2_hidden_next
        x = x + rnn2_hidden

        # Project Mels
        mels = self.mel_proj(x)
        mels = mels.view(batch_size, self.n_mels, self.max_r)[:, :, :self.r]
        hidden_states = (attn_hidden, rnn1_hidden, rnn2_hidden)
        cell_states = (rnn1_cell, rnn2_cell)

        return mels, scores, hidden_states, cell_states, context_vec


class Tacotron(nn.Module):
    def __init__(self, embed_dims, num_chars, encoder_dims, decoder_dims, n_mels, fft_bins, postnet_dims,
                 encoder_K, lstm_dims, postnet_K, num_highways, dropout, stop_threshold, num_speakers, speaker_emb_dim=128, smooth_factor=10.):
        super().__init__()
        self.n_mels = n_mels

        self.lstm_dims = lstm_dims
        self.decoder_dims = decoder_dims + speaker_emb_dim
        self.encoder = Encoder(embed_dims, num_chars, encoder_dims,
                               encoder_K, num_highways, dropout)
        self.encoder_proj = nn.Linear(self.decoder_dims, self.decoder_dims, bias=False)
        self.decoder = Decoder(n_mels, self.decoder_dims, lstm_dims, smooth_factor=smooth_factor)
        self.postnet = CBHG(postnet_K, n_mels, postnet_dims, [256, 80], num_highways)
        self.post_proj = nn.Linear(postnet_dims * 2, fft_bins, bias=False)

        self.init_model()
        self.num_params()

        self.register_buffer('step', torch.zeros(1, dtype=torch.long))
        self.register_buffer('stop_threshold', torch.tensor(stop_threshold, dtype=torch.float32))


    @property
    def r(self):
        return self.decoder.r.item()

    @r.setter
    def r(self, value):
        self.decoder.r = self.decoder.r.new_tensor(value, requires_grad=False)

    def forward(self, x, m, s_emb, generate_gta=False):
        device = next(self.parameters()).device  # use same device as parameters

        if self.training:
            self.step += 1

        batch_size, _, steps  = m.size()

        # Initialise all hidden states and pack into tuple
        attn_hidden = torch.zeros(batch_size, self.decoder_dims, device=device)
        rnn1_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        hidden_states = (attn_hidden, rnn1_hidden, rnn2_hidden)

        # Initialise all lstm cell states and pack into tuple
        rnn1_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        cell_states = (rnn1_cell, rnn2_cell)

        # <GO> Frame for start of decoder loop
        go_frame = torch.zeros(batch_size, self.n_mels, device=device)

        # Need an initial context vector
        context_vec = torch.zeros(batch_size, self.decoder_dims, device=device)

        # Project the encoder outputs to avoid
        # unnecessary matmuls in the decoder loop
        encoder_seq = self.encoder(x)
        speaker_emb = s_emb[:, None, :]
        speaker_emb = speaker_emb.repeat(1, encoder_seq.shape[1], 1)
        encoder_seq = torch.cat([encoder_seq, speaker_emb], dim=2)
        encoder_seq_proj = self.encoder_proj(encoder_seq)

        # Need a couple of lists for outputs
        mel_outputs, attn_scores = [], []

        # Run the decoder loop
        for t in range(0, steps, self.r):
            prenet_in = m[:, :, t - 1] if t > 0 else go_frame
            mel_frames, scores, hidden_states, cell_states, context_vec = \
                self.decoder(encoder_seq, encoder_seq_proj, prenet_in,
                             hidden_states, cell_states, context_vec, t)
            mel_outputs.append(mel_frames)
            attn_scores.append(scores)

        # Concat the mel outputs into sequence
        mel_outputs = torch.cat(mel_outputs, dim=2)

        # Post-Process for Linear Spectrograms
        postnet_out = self.postnet(mel_outputs)
        linear = self.post_proj(postnet_out)
        linear = linear.transpose(1, 2)

        # For easy visualisation
        attn_scores = torch.cat(attn_scores, 1)
        # attn_scores = attn_scores.cpu().data.numpy()

        return mel_outputs, linear, attn_scores

    def generate(self, x, s_emb, steps=2000):
        self.eval()
        device = next(self.parameters()).device  # use same device as parameters

        batch_size = 1
        x = torch.as_tensor(x, dtype=torch.long, device=device).unsqueeze(0)
        s_emb = torch.as_tensor(s_emb, dtype=torch.float, device=device).unsqueeze(0)

        # Need to initialise all hidden states and pack into tuple for tidyness
        attn_hidden = torch.zeros(batch_size, self.decoder_dims, device=device)
        rnn1_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        hidden_states = (attn_hidden, rnn1_hidden, rnn2_hidden)

        # Need to initialise all lstm cell states and pack into tuple for tidyness
        rnn1_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        cell_states = (rnn1_cell, rnn2_cell)

        # Need a <GO> Frame for start of decoder loop
        go_frame = torch.zeros(batch_size, self.n_mels, device=device)

        # Need an initial context vector
        context_vec = torch.zeros(batch_size, self.decoder_dims, device=device)

        # Project the encoder outputs to avoid
        # unnecessary matmuls in the decoder loop
        encoder_seq = self.encoder(x)
        speaker_emb = s_emb[:, None, :]
        speaker_emb = speaker_emb.repeat(1, encoder_seq.shape[1], 1)
        encoder_seq = torch.cat([encoder_seq, speaker_emb], dim=2)
        encoder_seq_proj = self.encoder_proj(encoder_seq)

        # Need a couple of lists for outputs
        mel_outputs, attn_scores = [], []

        # Run the decoder loop
        for t in range(0, steps, self.r):
            prenet_in = mel_outputs[-1][:, :, -1] if t > 0 else go_frame
            mel_frames, scores, hidden_states, cell_states, context_vec = \
            self.decoder(encoder_seq, encoder_seq_proj, prenet_in,
                         hidden_states, cell_states, context_vec, t)
            mel_outputs.append(mel_frames)
            attn_scores.append(scores)
            # Stop the loop if silent frames present
            if (mel_frames < self.stop_threshold).all() and t > 10: break

        # Concat the mel outputs into sequence
        mel_outputs = torch.cat(mel_outputs, dim=2)

        # Post-Process for Linear Spectrograms
        postnet_out = self.postnet(mel_outputs)
        linear = self.post_proj(postnet_out)
        linear = linear.transpose(1, 2)[0].cpu().data.numpy()
        mel_outputs = mel_outputs[0].cpu().data.numpy()

        # For easy visualisation
        attn_scores = torch.cat(attn_scores, 1)
        attn_scores = attn_scores.cpu().data.numpy()[0]

        self.train()

        return mel_outputs, linear, attn_scores

    def init_model(self):
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)

    def get_step(self):
        return self.step.data.item()

    def reset_step(self):
        # assignment to parameters or buffers is overloaded, updates internal dict entry
        self.step = self.step.data.new_tensor(1)

    def log(self, path, msg):
        with open(path, 'a') as f:
            print(msg, file=f)

    def load(self, path: Union[str, Path]):
        # Use device of model params as location for loaded state
        device = next(self.parameters()).device
        state_dict = torch.load(path, map_location=device)

        # Backwards compatibility with old saved models
        if 'r' in state_dict and not 'decoder.r' in state_dict:
            self.r = state_dict['r']

        self.load_state_dict(state_dict, strict=False)

    def save(self, path: Union[str, Path]):
        # No optimizer argument because saving a model should not include data
        # only relevant in the training process - it should only be properties
        # of the model itself. Let caller take care of saving optimzier state.
        torch.save(self.state_dict(), path)

    def num_params(self, print_out=True):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        if print_out:
            print('Trainable Parameters: %.3fM' % parameters)
        return parameters
