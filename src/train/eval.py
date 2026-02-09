"""
Evaluation helpers.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def encode_to_latent(model, x_test, device, batch_size: int = 1024):
    """
    Works for:
      - CondGRUModel: has model.enc
      - EncoderOnly: forward(x) returns v (and also has .enc)
    """
    model.eval()

    if not torch.is_tensor(x_test):
        x_test = torch.tensor(x_test, dtype=torch.float32)

    x_test = x_test.to(device)
    n = x_test.shape[0]
    vs = []

    for start in range(0, n, batch_size):
        xb = x_test[start : start + batch_size]
        if hasattr(model, "enc"):
            vb = model.enc(xb)
        else:
            vb = model(xb)
        vs.append(vb.detach().cpu())

    return torch.cat(vs, dim=0).numpy()
