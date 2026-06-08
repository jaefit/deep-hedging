"""Neural hedging policy.

A single MLP shared across all time steps maps the local state to a share
holding. State features per step: log-moneyness log(S/K), time-to-maturity
fraction tau/T, and the current holding h_prev (so the net can learn to trade
*incrementally* and economize on transaction costs). Output is squashed to
[0, 1] — the natural range of a short-call delta hedge.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class HedgePolicy(nn.Module):
    def __init__(self, hidden: int = 32, n_features: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, feats):  # feats: (B, n_features)
        return torch.sigmoid(self.net(feats)).squeeze(-1)  # (B,) in [0,1]
