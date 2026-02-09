"""
Modular model components for conditional sequence prediction.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    def __init__(
        self,
        d_x: int = 2,
        d_v: int = 8,
        hidden: int = 128,
        output_activation: Optional[str] = "sigmoid",
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_x, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_v),
        )
        if output_activation is None:
            self.out_act = None
        elif output_activation == "sigmoid":
            self.out_act = torch.sigmoid
        else:
            raise ValueError("output_activation must be None or 'sigmoid'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.net(x)
        return self.out_act(v) if self.out_act is not None else v


class MLPDecoder(nn.Module):
    def __init__(self, d_h: int = 64, d_o: int = 1, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_h, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_o),
        )

    def forward(self, h_seq: torch.Tensor) -> torch.Tensor:
        return self.net(h_seq)


class RNNBackbone(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        rnn_type: str = "gru",
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        rnn_type = rnn_type.lower()
        if rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        elif rnn_type == "rnn":
            self.rnn = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                nonlinearity="tanh",
                dropout=dropout if num_layers > 1 else 0.0,
            )
        else:
            raise ValueError("rnn_type must be 'gru' or 'rnn'")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.rnn(x)


class CondRNNModel(nn.Module):
    def __init__(self, encoder: nn.Module, rnn: nn.Module, decoder: nn.Module) -> None:
        super().__init__()
        self.enc = encoder
        self.rnn = rnn
        self.dec = decoder

    def forward(self, x: torch.Tensor, o_hist: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x:      (B, d_x)
        o_hist: (B, L, d_o)
        returns o_hat_next: (B, L, d_o), v: (B, d_v)
        """
        v = self.enc(x)
        bsz, seq_len, _ = o_hist.shape
        v_rep = v[:, None, :].expand(bsz, seq_len, v.shape[-1])
        inp = torch.cat([o_hist, v_rep], dim=-1)
        h_seq, _ = self.rnn(inp)
        o_hat_next = self.dec(h_seq)
        return o_hat_next, v


# Clearer alias name for readability.
ConditionalSequenceModel = CondRNNModel


class EncoderOnly(nn.Module):
    def __init__(self, encoder_module: nn.Module) -> None:
        super().__init__()
        self.enc = encoder_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)


def build_cond_gru_model(
    d_x: int = 2,
    d_o: int = 1,
    d_v: int = 8,
    d_h: int = 64,
    hidden_enc: int = 128,
    hidden_dec: int = 128,
    output_activation: Optional[str] = "sigmoid",
) -> CondRNNModel:
    encoder = MLPEncoder(d_x=d_x, d_v=d_v, hidden=hidden_enc, output_activation=output_activation)
    rnn = RNNBackbone(input_size=d_o + d_v, hidden_size=d_h, rnn_type="gru")
    decoder = MLPDecoder(d_h=d_h, d_o=d_o, hidden=hidden_dec)
    return CondRNNModel(encoder=encoder, rnn=rnn, decoder=decoder)
