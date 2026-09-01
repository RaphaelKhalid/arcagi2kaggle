"""Symbolic output-property predictors for ARC-AGI-2.

High-precision, falsifiable predictors of test-output grid size and color
palette, intended to prune a neural solver's decode search space. Every rule
must explain ALL demonstration pairs exactly to fire, and the combined
predictors abstain on any disagreement between fired rules.

    from symbolic import predict_size, predict_palette, palette_superset_bound

Measurement: python -m symbolic.measure
"""

from .palette_predictor import palette_superset_bound, predict_palette
from .size_predictor import predict_size

__all__ = ["predict_size", "predict_palette", "palette_superset_bound"]
