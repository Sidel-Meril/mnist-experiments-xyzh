"""Shared data and modeling contracts.

Experiment YAML files stay next to each package. Pass that path to ``load``.
"""

from mnist_contracts.data_contract import DataContract
from mnist_contracts.model_contract import ModelingContract

__all__ = ["DataContract", "ModelingContract"]
