"""
Condition encoder for Mel spectrograms with optional demographic fusion.
"""
import torch
import torch.nn as nn
from .demographic_encoder import DemographicEncoder
try:
    from .enhanced_demographic_encoder import EnhancedDemographicEncoder
    ENHANCED_ENCODER_AVAILABLE = True
except ImportError:
    ENHANCED_ENCODER_AVAILABLE = False
    EnhancedDemographicEncoder = None


class ConditionEncoder(nn.Module):
    """
    Encodes Mel spectrogram (and optional demographic features) into condition vector.

    Architecture:
        Mel Input: (B, 60, 128) or (B, 1, 60, 128)
        Conv2D layers for Mel + optional demographic encoder
        Output: (B, d_c) where d_c=condition_dim
    """

    def __init__(
        self,
        input_channels=1,
        hidden_dims=None,
        condition_dim=256,
        demographic_dim=64,
        use_demographic=True,
        demographic_input_dim=4,
        use_enhanced_encoder=True,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [32, 64, 128]

        self.input_channels = input_channels
        self.hidden_dims = hidden_dims
        self.condition_dim = condition_dim
        self.use_demographic = bool(use_demographic)
        self.demographic_input_dim = int(demographic_input_dim)
        self.demographic_dim = int(demographic_dim)
        self.use_enhanced_encoder = bool(use_enhanced_encoder) and ENHANCED_ENCODER_AVAILABLE

        layers = []
        in_channels = input_channels
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            ])
            in_channels = hidden_dim
        self.conv_layers = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        if self.use_demographic and self.demographic_input_dim > 0:
            mel_feature_dim = condition_dim - self.demographic_dim
            self.mel_projection = nn.Sequential(
                nn.Linear(hidden_dims[-1], mel_feature_dim),
                nn.LayerNorm(mel_feature_dim),
            )

            if self.use_enhanced_encoder:
                self.demographic_encoder = EnhancedDemographicEncoder(
                    input_dim=self.demographic_input_dim,
                    demographic_dim=self.demographic_dim,
                    use_interaction=True,
                    use_attention=True,
                )
            else:
                self.demographic_encoder = DemographicEncoder(
                    input_dim=self.demographic_input_dim,
                    demographic_dim=self.demographic_dim,
                    use_normalization=True,
                )

            self.fusion = nn.Sequential(
                nn.Linear(condition_dim, condition_dim),
                nn.LayerNorm(condition_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
            )
        else:
            self.mel_projection = nn.Sequential(
                nn.Linear(hidden_dims[-1], condition_dim),
                nn.LayerNorm(condition_dim),
            )

    def forward(self, mel, demographic=None):
        """
        Encode Mel spectrogram (and optional demographic) to condition vector.

        Args:
            mel: Tensor of shape (B, 60, 128) or (B, 1, 60, 128)

        Returns:
            Condition vector of shape (B, condition_dim)
        """
        if mel.dim() == 3:
            mel = mel.unsqueeze(1)
        feat = self.conv_layers(mel)
        x = self.global_pool(feat).flatten(1)

        mel_features = self.mel_projection(x)

        if self.use_demographic and (self.demographic_input_dim > 0) and (demographic is not None):
            demo_features = self.demographic_encoder(demographic)
            combined = torch.cat([mel_features, demo_features], dim=1)
            return self.fusion(combined)

        return mel_features
