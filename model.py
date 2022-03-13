import math
from types import ModuleType
from typing import Tuple

import torch
import torch.nn as nn
from torchtyping import TensorType


class PositionalEncoding(nn.Module):
    """
    Positional encoding in Transformer fashion:
    with a use of goniometric functions
    """
    def __init__(self, input_size: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, input_size, 2) * (-math.log(10000.0) / input_size))
        pe = torch.zeros(max_len, 1, input_size)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, input: TensorType) -> TensorType:
        input = input + self.pe[:input.size(0)]
        return self.dropout(input)


class LSTMModel(nn.Module):
    """
    LSTM model with embedding lookup table, positional encoding,
    LSTM core and decoder which transforms LSTM outputs to predictions.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        seq_len: int,
        num_embeddings: int
        ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=input_size)
        self.pos_encoder = PositionalEncoding(input_size=input_size)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size)
        self.decoder = nn.Linear(hidden_size * seq_len, 1)

        self.num_layers = self.lstm.num_layers
        self.hidden_size = hidden_size

    def init_hidden(self, batch_size: int) -> Tuple[TensorType]:
        """
        Hidden state of LSTM needs to be reinitialized after each batch, otherwise
        it will contain informations from the sequences from the previous batch.
        """
        hx = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        cx = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return (hx, cx)

    def forward(self, input: TensorType) -> TensorType:
        input = input["source"]
        batch_size = input.shape[0]
        emb_inp = self.embedding(input)
        posenc_inp = self.pos_encoder(emb_inp)
        hx, cx = self.lstm(posenc_inp)
        hx = hx.reshape(batch_size, -1)
        output = self.decoder(hx)
        return output


class TransformerModel(nn.Module):
    """
    Transformer model with embedding lookup table, positional encoding,
    Transformer core and decoder which transforms LSTM outputs
    to predictions.
    """
    def __init__(
        self,
        input_size: int,
        num_embeddings: int,
        hidden_size: int,
        nhead: int,
        nlayer: int,
        seq_len: int
        ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=input_size)
        self.pos_encoder = PositionalEncoding(input_size=input_size)
        self.encoder = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.transformer = nn.Transformer(
            d_model=hidden_size,
            nhead=nhead,
            num_encoder_layers=nlayer,
            num_decoder_layers=nlayer
            )
        self.decoder = nn.Linear(hidden_size * seq_len, 1)
        
    def init_hidden(self, *args, **kwargs):
        """
        Here to be compatible with training module. Transformer does not
        have hidden state in the same fashion as LSTM so reinitialization
        is not needed.
        """
        pass

    def forward(self, input: TensorType) -> TensorType:
        input = input["source"]
        batch_size = input.shape[0]
        emb_inp = self.embedding(input)
        posenc_inp = self.pos_encoder(emb_inp)
        enc_inp = self.encoder(posenc_inp)
        target = enc_inp
        transformer_output = self.transformer(enc_inp, target)
        transformer_output = transformer_output.reshape(batch_size, -1)
        output = self.decoder(transformer_output)
        return output
