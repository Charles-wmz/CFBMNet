"""
Enhanced demographic encoder with feature interaction and attention mechanisms.
Originally optimized for gender + height + weight, now supports variable feature dims.
"""
import torch
import torch.nn as nn


class EnhancedDemographicEncoder(nn.Module):
    """
    Enhanced demographic encoder with feature interaction and attention.
    
    Architecture:
        1. Individual feature encoding (each scalar feature separately)
        2. Feature interaction encoding (pairwise cross features)
        3. Self-attention for feature importance
        4. Final projection to demographic_dim
    
    Args:
        input_dim: Number of input features (supports 1+ for ablations)
        demographic_dim: Output dimension (default 64)
        use_interaction: Whether to use feature interaction (default True)
        use_attention: Whether to use attention mechanism (default True)
    """
    
    def __init__(
        self, 
        input_dim=3, 
        demographic_dim=64,
        use_interaction=True,
        use_attention=True
    ):
        super().__init__()
        
        self.input_dim = int(input_dim)
        self.demographic_dim = int(demographic_dim)
        self.use_interaction = bool(use_interaction) and (self.input_dim >= 2)
        self.use_attention = bool(use_attention)
        
        # Normalization
        self.normalize = nn.BatchNorm1d(self.input_dim)
        
        # Individual feature encoders
        # Each scalar feature gets its own small MLP (shared architecture, independent weights).
        def _make_scalar_encoder():
            return nn.Sequential(
                nn.Linear(1, 16),
                nn.LayerNorm(16),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(16, 32),
            )

        self.feature_encoders = nn.ModuleList([_make_scalar_encoder() for _ in range(self.input_dim)])
        
        # Feature interaction encoder
        if self.use_interaction:
            # Encode all pairwise interactions x_i * x_j
            num_pairs = (self.input_dim * (self.input_dim - 1)) // 2
            self.interaction_encoder = nn.Sequential(
                nn.Linear(num_pairs, 32),
                nn.LayerNorm(32),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(32, 32)
            )
        
        # Self-attention for feature importance
        if self.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=32,
                num_heads=4,
                batch_first=True,
                dropout=0.1
            )
        
        # Calculate final input dimension
        final_input_dim = 32 * self.input_dim
        if self.use_interaction:
            final_input_dim += 32
        if self.use_attention:
            final_input_dim += 32
        
        # Final projection
        self.final_proj = nn.Sequential(
            nn.Linear(final_input_dim, demographic_dim * 2),
            nn.LayerNorm(demographic_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(demographic_dim * 2, demographic_dim),
            nn.LayerNorm(demographic_dim)
        )
    
    def forward(self, demographic):
        """
        Forward pass.
        
        Args:
            demographic: Tensor of shape (B, input_dim)
        
        Returns:
            Demographic feature vector of shape (B, demographic_dim)
        """
        # Normalize input
        demographic = self.normalize(demographic)
        
        # Encode each scalar feature independently
        scalar_feats = [demographic[:, i:i+1] for i in range(self.input_dim)]  # each (B, 1)
        encoded_feats = [enc(x) for enc, x in zip(self.feature_encoders, scalar_feats)]  # each (B, 32)

        # Collect features for fusion
        features_list = list(encoded_feats)
        
        # Feature interaction
        if self.use_interaction:
            # Compute all pairwise products: (B, C(input_dim, 2))
            pairs = []
            for i in range(self.input_dim):
                for j in range(i + 1, self.input_dim):
                    pairs.append(scalar_feats[i] * scalar_feats[j])
            interactions = torch.cat(pairs, dim=1) if len(pairs) > 0 else demographic.new_zeros((demographic.size(0), 0))
            interaction_feat = self.interaction_encoder(interactions)  # (B, 32)
            features_list.append(interaction_feat)
        
        # Self-attention for feature importance
        if self.use_attention:
            # Stack features for attention: (B, input_dim, 32)
            feature_stack = torch.stack(encoded_feats, dim=1)
            
            # Self-attention
            attended, _attention_weights = self.attention(
                feature_stack, feature_stack, feature_stack
            )  # (B, input_dim, 32)
            
            # Aggregate attended features (mean pooling)
            attended_feat = attended.mean(dim=1)  # (B, 32)
            features_list.append(attended_feat)
        
        # Concatenate all features
        combined = torch.cat(features_list, dim=1)  # (B, final_input_dim)
        
        # Final projection
        output = self.final_proj(combined)  # (B, demographic_dim)
        
        return output

