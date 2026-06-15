"""Model factory for the Curve-First Basis-Mixture Network."""
import torch.nn as nn

from .regression_model import DirectFlowRegression


def model_name(config) -> str:
    return str(getattr(config, "MODEL", "direct")).lower().strip()


def create_torch_model(config) -> nn.Module:
    name = model_name(config)
    if name == "direct":
        return DirectFlowRegression.create_model(config)
    raise ValueError(f"Unknown MODEL={name!r}. Expected: direct.")


def count_torch_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def count_trainable_torch_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
