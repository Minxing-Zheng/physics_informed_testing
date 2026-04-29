"""
Modular model components for conditional sequence prediction.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class SpectralConv1d(nn.Module):
    """
    1D spectral convolution used in Fourier neural operator style blocks.

    Input/Output shape: (B, C, L)
    """

    def __init__(self, in_channels: int, out_channels: int, n_modes: int) -> None:
        super().__init__()
        if n_modes < 1:
            raise ValueError("n_modes must be >= 1")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.n_modes = int(n_modes)

        scale = 1.0 / max(1, in_channels * out_channels)
        self.weight = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, self.n_modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected x with shape (B, C, L), got {tuple(x.shape)}")
        bsz, _, seq_len = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        n_freq = x_ft.shape[-1]
        n_used = min(self.n_modes, n_freq)

        out_ft = torch.zeros(
            bsz,
            self.out_channels,
            n_freq,
            device=x.device,
            dtype=torch.cfloat,
        )
        out_ft[:, :, :n_used] = torch.einsum(
            "bcm,com->bom",
            x_ft[:, :, :n_used],
            self.weight[:, :, :n_used],
        )
        return torch.fft.irfft(out_ft, n=seq_len, dim=-1)


class NeuralOperatorBlock1d(nn.Module):
    """
    Single operator block mixing global spectral interactions and local channels.
    """

    def __init__(self, width: int, n_modes: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.spec = SpectralConv1d(width, width, n_modes=n_modes)
        self.local = nn.Conv1d(width, width, kernel_size=1)
        self.norm = nn.BatchNorm1d(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.spec(x) + self.local(x)
        y = self.norm(y)
        y = F.gelu(y)
        return self.dropout(y)


class NeuralOperatorBackbone(nn.Module):
    """
    Conditional 1D neural-operator backbone over the time axis.

    Input:  (B, L, d_o + d_v)
    Output: (B, L, d_h)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 4,
        n_modes: int = 16,
        width: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        width_eff = int(hidden_size if width is None else width)
        if width_eff < 1:
            raise ValueError("width must be >= 1")
        self.input_proj = nn.Linear(input_size, width_eff)
        self.blocks = nn.ModuleList(
            [
                NeuralOperatorBlock1d(width=width_eff, n_modes=n_modes, dropout=dropout)
                for _ in range(int(num_layers))
            ]
        )
        self.output_proj = nn.Linear(width_eff, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected x with shape (B, L, C), got {tuple(x.shape)}")
        h = self.input_proj(x)  # (B, L, width)
        h = h.transpose(1, 2)  # (B, width, L)
        for block in self.blocks:
            h = h + block(h)  # residual operator update
        h = h.transpose(1, 2)  # (B, L, width)
        return self.output_proj(h)  # (B, L, d_h)


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


class CondNeuralOperatorModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        operator: nn.Module,
        decoder: nn.Module,
        residual_update: bool = True,
        residual_dt: float = 1.0,
    ) -> None:
        super().__init__()
        self.enc = encoder
        self.operator = operator
        self.dec = decoder
        self.residual_update = bool(residual_update)
        self.residual_dt = float(residual_dt)

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
        h_seq = self.operator(inp)
        delta_or_out = self.dec(h_seq)
        if self.residual_update:
            o_hat_next = o_hist + self.residual_dt * delta_or_out
        else:
            o_hat_next = delta_or_out
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


def build_neural_operator_model(
    d_x: int = 2,
    d_o: int = 1,
    d_v: int = 8,
    d_h: int = 64,
    hidden_enc: int = 128,
    hidden_dec: int = 128,
    enc_layers: int = 2,
    dec_layers: int = 1,
    output_activation: Optional[str] = "sigmoid",
    operator_layers: int = 4,
    operator_modes: int = 16,
    operator_width: Optional[int] = None,
    dropout_operator: float = 0.0,
    residual_update: bool = True,
    residual_dt: float = 1.0,
) -> CondNeuralOperatorModel:
    """
    Build a conditional neural-operator model with the same train/eval API as
    build_cond_gru_model.
    """
    encoder = MLPEncoder(
        d_x=d_x,
        d_v=d_v,
        hidden_size=hidden_enc,
        num_hidden_layers=enc_layers,
        output_activation=output_activation,
    )
    operator = NeuralOperatorBackbone(
        input_size=d_o + d_v,
        hidden_size=d_h,
        num_layers=operator_layers,
        n_modes=operator_modes,
        width=operator_width,
        dropout=dropout_operator,
    )
    decoder = MLPDecoder(d_h=d_h, d_o=d_o, hidden_size=hidden_dec, num_hidden_layers=dec_layers)
    return CondNeuralOperatorModel(
        encoder=encoder,
        operator=operator,
        decoder=decoder,
        residual_update=residual_update,
        residual_dt=residual_dt,
    )
