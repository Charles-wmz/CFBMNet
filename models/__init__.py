"""Models package for CFBMNet flow curve prediction."""
from .condition_encoder import ConditionEncoder
from .demographic_encoder import DemographicEncoder
from .regression_model import DirectFlowRegression
from . import model_registry

try:
    from .enhanced_demographic_encoder import EnhancedDemographicEncoder
    __all__ = [
        'ConditionEncoder',
        'DemographicEncoder',
        'EnhancedDemographicEncoder',
        'DirectFlowRegression',
        'model_registry',
    ]
except ImportError:
    __all__ = [
        'ConditionEncoder',
        'DemographicEncoder',
        'DirectFlowRegression',
        'model_registry',
    ]
