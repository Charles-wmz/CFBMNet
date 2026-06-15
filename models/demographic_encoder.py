"""
Demographic encoder for population statistics information.
"""
import torch.nn as nn


class DemographicEncoder(nn.Module):
    """
    Encodes demographic information (gender, age, height, weight) into feature vector.
    Supports variable input dimensions for ablation studies.
    
    Architecture:
        Input: (B, input_dim) where input_dim can be 1-4
        Normalization and embedding layers
        Output: (B, demographic_dim)
    
    Args:
        input_dim: Number of input demographic features (1-4)
        demographic_dim: Dimension of output demographic feature vector
        use_normalization: Whether to normalize input features
    """
    
    def __init__(self, input_dim=4, demographic_dim=64, use_normalization=True):
        """Initialize demographic encoder."""
        super().__init__()
        
        self.input_dim = input_dim
        self.demographic_dim = demographic_dim
        self.use_normalization = use_normalization
        
        # Feature normalization (learnable)
        if use_normalization:
            self.normalize = nn.BatchNorm1d(input_dim)
        
        # MLP to encode demographic features
        # Adjust hidden dimensions based on input size
        if input_dim == 1:
            # Single feature: simpler architecture
            hidden_dims = [16, 32]
        elif input_dim == 2:
            hidden_dims = [24, 48]
        elif input_dim == 3:
            hidden_dims = [32, 64]
        elif input_dim == 4:
            hidden_dims = [32, 64]
        else:  # input_dim >= 5 (e.g., gender + height + weight + BMI)
            hidden_dims = [32, 64]
        
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            ])
            in_dim = hidden_dim
        
        # Final projection to demographic_dim
        layers.extend([
            nn.Linear(in_dim, demographic_dim),
            nn.LayerNorm(demographic_dim)
        ])
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, demographic):
        """
        Encode demographic information to feature vector.
        
        Args:
            demographic: Tensor of shape (B, input_dim)
        
        Returns:
            Demographic feature vector of shape (B, demographic_dim)
        """
        # Normalize if enabled
        if self.use_normalization:
            demographic = self.normalize(demographic)
        
        # Encode to feature vector
        demo_features = self.encoder(demographic)  # (B, demographic_dim)
        
        return demo_features

