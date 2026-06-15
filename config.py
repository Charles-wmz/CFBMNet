"""
Configuration for respiratory flow-curve prediction.
This file keeps fixed defaults for model structure, training hyperparameters, and loss weights.
Runtime switches controlled by train_CV.py, such as innovation modules, demographics, and device
selection, are assigned to the Config class in train_CV.py main().
"""
import json
from typing import Any, Dict


class Config:
    """Global configuration for CFBMNet training and evaluation."""
    
    # Model name kept for registry compatibility.
    MODEL = 'direct'
    
    # Data paths
    MEL_DIR = 'data_pre/mel'
    CSV_DIR = 'data_pre/csv'
    RAW_META_DIR = 'data_pre/meta'
    LABEL_FILE = 'data_pre/label.csv'
    
    # Data parameters
    SEQUENCE_LENGTH = 60  # Number of time points over the 0-3 s window.
    SAMPLE_RATE = 48000  # Audio sampling rate.
    
    # Mel spectrogram parameters, aligned with preprocess_data.py.
    N_FFT = 2048
    HOP_LENGTH = 2400  # 48000 * 3 / 60 = 2400
    N_MELS = 128
    
    # Model structure: condition encoder plus regression head.
    CONDITION_DIM = 256
    CONDITION_ENCODER_HIDDEN_DIMS = [64, 128, 256]  # Hidden dimensions for the condition encoder.
    
    # Demographic embedding dimension, fixed inside the model.
    DEMOGRAPHIC_DIM = 64
    
    # Training parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0002
    NUM_EPOCHS = 300
    WEIGHT_DECAY = 5e-5
    EARLY_STOPPING_PATIENCE = 30  # Stop if validation loss does not improve for this many epochs.
    SPEC_AUGMENT = True  # Apply SpecAugment to Mel spectrograms during training.
    
    # Regression loss configuration
    # Total Loss = LAMBDA_REG × Loss_Reg [+ LAMBDA_PHYS × Loss_Phys + LAMBDA_SMOOTH × Loss_Smooth]
    # Loss_Phys = FEV1_WEIGHT × Loss_FEV1 + FVC_WEIGHT × Loss_FVC + PEF_WEIGHT × Loss_PEF
    LAMBDA_REG = 1.0  # Main curve regression loss weight.
    LAMBDA_PHYS = 1.0  # Overall physics constraint loss weight.
    FEV1_WEIGHT = 1.0  # FEV1 component weight.
    FVC_WEIGHT = 1.0   # FVC component weight.
    PEF_WEIGHT = 1.0   # PEF component weight.
    LAMBDA_SMOOTH = 0.1  # Smoothness loss weight; set to 0 to disable.
    # Default Sobolev-style regression loss parameters, overridable from train_CV.py.
    LOSS_TYPE = 'l1'  # Default regression loss; Innovation 3 switches this to sobolev_huber.
    SOB_ALPHA = 0.2  # First-derivative term weight.
    SOB_BETA = 0.05  # Second-derivative term weight.
    SOB_DELTA = 1.0  # Huber threshold.
    
    # On Windows and small datasets, 0 workers avoids multiprocessing overhead.
    NUM_WORKERS = 0
    
    # Random seed for reproducibility.
    RANDOM_SEED = 42

    # Regression head hyperparameters.
    REGRESSOR_HIDDEN_DIM = 256
    REGRESSOR_DROPOUT = 0.1
    REGRESSOR_NUM_LAYERS = 1  # Number of MLP hidden layers, excluding the output layer.
    
    # Gradient clipping.
    GRAD_CLIP_NORM = 1.0  # Maximum gradient norm used to reduce exploding gradients.
    
    @classmethod
    def to_dict(cls):
        """Convert the configuration class to a plain dictionary."""
        return {
            key: value for key, value in cls.__dict__.items()
            if not key.startswith('_') and not callable(value)
        }
    
    @classmethod
    def load_json(cls, json_path: str) -> Dict[str, Any]:
        """
        Load a configuration dictionary from a JSON file.
        
        Args:
            json_path: Path to the JSON configuration file.
        
        Returns:
            Configuration dictionary.
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @classmethod
    def save_to_json(cls, json_path: str):
        """
        Save the current configuration to a JSON file.
        
        Args:
            json_path: Output JSON path.
        """
        config_dict = cls.to_dict()
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def update_from_dict(cls, config_dict: Dict[str, Any]):
        """
        Update class attributes from a dictionary.
        
        Args:
            config_dict: Configuration key-value pairs to update.
        """
        for key, value in config_dict.items():
            if hasattr(cls, key) or not key.startswith('_'):
                setattr(cls, key, value)
    
    @classmethod
    def update_from_args(cls, args):
        """
        Update class attributes from parsed command-line arguments.
        
        Args:
            args: argparse.Namespace object.
        """
        config_dict = {}
        for key, value in vars(args).items():
            if value is not None and key != 'config_file':  # Skip config_file as it's already handled
                # Convert to uppercase and replace hyphens with underscores to match class attributes
                upper_key = key.upper().replace('-', '_')
                if hasattr(cls, upper_key):
                    config_dict[upper_key] = value
        
        if config_dict:
            cls.update_from_dict(config_dict)
