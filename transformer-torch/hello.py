import math

import torch
import torch.nn as nn


class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # create a matrix of shape(seq_len,d_model)
        pe = torch.zeros(seq_len, d_model)
        # create a vector of shape(seq_len,1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self, features, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    "This consists of two linear transformations with a ReLU activation in between. `FFN⁢(x)=max⁡(0,x⁢W1+b1)⁢W2+b2`"

    def __init__(self, d_model: int, dff: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.dff = dff
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(d_model, dff)  # W1 and B1
        self.linear_2 = nn.Linear(dff, d_model)  # W2 and B2

    def forward(self, x):
        "input (batch,seq_len,d_model) --> (batch,seq_len,dff) -->output (batch,seq_len,d_model)"
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
