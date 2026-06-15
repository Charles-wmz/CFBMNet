"""
Direct discriminative regression model for flow curve prediction.
"""
from typing import Optional

import torch
import torch.nn as nn

from .condition_encoder import ConditionEncoder


class BasisMixtureCurveDecoder(nn.Module):
    """
    Innovation 1: Basis-Mixture Curve Decoder.

    Decomposes curve generation into a basis mixture plus a lightweight residual:
    - Learn a curve basis bank B = {b_1,...,b_K}, where b_k is in R^{seq_len}.
    - Predict coefficients alpha(c) from the condition vector c to form y_base.
    - Predict a residual delta_y(c), producing y = y_base + delta_y(c).
    """

    def __init__(self, condition_dim: int, seq_len: int = 60, hidden_dim: int = 256, num_bases: int = 8):
        super().__init__()
        self.condition_dim = int(condition_dim)
        self.seq_len = int(seq_len)
        self.hidden_dim = int(hidden_dim)
        self.num_bases = int(num_bases)

        # Learnable curve bases, each with length seq_len.
        self.basis_bank = nn.Parameter(torch.randn(self.num_bases, self.seq_len))

        # Coefficient prediction: alpha(c).
        self.coeff_mlp = nn.Sequential(
            nn.Linear(self.condition_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.num_bases),
        )

        # Lightweight residual branch: delta_y(c).
        self.residual_mlp = nn.Sequential(
            nn.Linear(self.condition_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.seq_len),
        )

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            condition: (B, condition_dim)
        Returns:
            flow: (B, seq_len)
        """
        # Basis coefficients alpha(c). No softmax is imposed, allowing signed combinations.
        alpha = self.coeff_mlp(condition)  # (B,K)

        # y_base = Σ_k α_k b_k
        Bk = self.basis_bank.unsqueeze(0)  # (1,K,T)
        y_base = torch.matmul(alpha.unsqueeze(1), Bk).squeeze(1)  # (B,T)

        # Detail residual delta_y(c).
        delta_y = self.residual_mlp(condition)  # (B,T)

        return y_base + delta_y


class DirectFlowRegression(nn.Module):
    """
    Curve-first regression model: predict a 60-point flow curve from condition input.

    Input:
        mel: (B, 60, 128)
        demographic (optional): (B, demographic_input_dim)
    Output:
        flow_pred: (B, 60)
    """

    def __init__(self, condition_encoder: ConditionEncoder, regressor: nn.Module, temporal_regressor: Optional[nn.Module] = None):
        super().__init__()
        self.condition_encoder = condition_encoder
        self.regressor = regressor
        self.temporal_regressor = temporal_regressor

    def forward(self, mel, demographic=None):
        c = self.condition_encoder(mel, demographic)  # (B, condition_dim)
        if self.temporal_regressor is not None:
            return self.temporal_regressor(c)
        flow = self.regressor(c)  # (B, 60)
        return flow

    @staticmethod
    def create_model(config):
        use_demographic = bool(getattr(config, 'USE_DEMOGRAPHIC', False))
        demographic_features = getattr(config, 'DEMOGRAPHIC_FEATURES', ['gender', 'age', 'height', 'weight'])
        demographic_input_dim = len(demographic_features) if use_demographic else 0
        use_enhanced_encoder = bool(getattr(config, 'USE_ENHANCED_DEMOGRAPHIC_ENCODER', True))
        use_st_encoder = bool(getattr(config, 'USE_SPECTRO_TEMPORAL_CONDITION_ENCODER', False))

        condition_hidden_dims = getattr(config, 'CONDITION_ENCODER_HIDDEN_DIMS', [32, 64, 128])
        condition_dim = int(getattr(config, 'CONDITION_DIM', 256))

        condition_encoder = ConditionEncoder(
            input_channels=1,
            hidden_dims=condition_hidden_dims,
            condition_dim=condition_dim,
            demographic_dim=getattr(config, 'DEMOGRAPHIC_DIM', 64),
            use_demographic=use_demographic,
            demographic_input_dim=demographic_input_dim,
            use_enhanced_encoder=use_enhanced_encoder,
        )

        seq_len = int(getattr(config, 'SEQUENCE_LENGTH', 60))

        # Fallback MLP head used when the basis-mixture decoder is disabled.
        hidden = int(getattr(config, 'REGRESSOR_HIDDEN_DIM', 256))
        dropout = float(getattr(config, 'REGRESSOR_DROPOUT', 0.1))
        num_layers = int(getattr(config, 'REGRESSOR_NUM_LAYERS', 1))

        layers = []
        in_dim = condition_dim
        for _ in range(max(num_layers, 1)):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.LayerNorm(hidden))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, seq_len))
        regressor = nn.Sequential(*layers)

        temporal_regressor: Optional[nn.Module] = None
        if use_st_encoder:
            num_bases = int(getattr(config, 'BASIS_MIXTURE_NUM_BASES', 8))
            temporal_regressor = BasisMixtureCurveDecoder(
                condition_dim=condition_dim,
                seq_len=seq_len,
                hidden_dim=hidden,
                num_bases=num_bases,
            )

        return DirectFlowRegression(condition_encoder=condition_encoder, regressor=regressor, temporal_regressor=temporal_regressor)

