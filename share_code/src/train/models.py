"""
Modular model components for conditional sequence prediction.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def _build_mlp(
    in_dim: int,
    out_dim: int,
    hidden_size: int,
    num_hidden_layers: int,
    output_activation: Optional[str] = None,
) -> nn.Sequential:
    if num_hidden_layers < 1:
        raise ValueError("num_hidden_layers must be >= 1")
    layers = [nn.Linear(in_dim, hidden_size), nn.ReLU()]
    for _ in range(num_hidden_layers - 1):
        layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
    layers.append(nn.Linear(hidden_size, out_dim))
    net = nn.Sequential(*layers)
    if output_activation is None:
        out_act = None
    elif output_activation == "sigmoid":
        out_act = torch.sigmoid
    else:
        raise ValueError("output_activation must be None or 'sigmoid'")
    return net, out_act


class MLPEncoder(nn.Module):
    def __init__(
        self,
        d_x: int = 2,
        d_v: int = 8,
        hidden_size: int = 128,
        num_hidden_layers: int = 2,
        output_activation: Optional[str] = "sigmoid",
    ) -> None:
        super().__init__()
        self.net, self.out_act = _build_mlp(
            in_dim=d_x,
            out_dim=d_v,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            output_activation=output_activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.net(x)
        return self.out_act(v) if self.out_act is not None else v


class MLPDecoder(nn.Module):
    def __init__(
        self,
        d_h: int = 64,
        d_o: int = 1,
        hidden_size: int = 128,
        num_hidden_layers: int = 1,
    ) -> None:
        super().__init__()
        self.net, _ = _build_mlp(
            in_dim=d_h,
            out_dim=d_o,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            output_activation=None,
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
    enc_layers: int = 2,
    dec_layers: int = 1,
    output_activation: Optional[str] = "sigmoid",
    dropout_rnn = 0.1
) -> CondRNNModel:
    encoder = MLPEncoder(
        d_x=d_x,
        d_v=d_v,
        hidden_size=hidden_enc,
        num_hidden_layers=enc_layers,
        output_activation=output_activation,
    )
    rnn = RNNBackbone(input_size=d_o + d_v, hidden_size=d_h, rnn_type="gru",dropout=dropout_rnn)
    decoder = MLPDecoder(d_h=d_h, d_o=d_o, hidden_size=hidden_dec, num_hidden_layers=dec_layers)
    return CondRNNModel(encoder=encoder, rnn=rnn, decoder=decoder)
