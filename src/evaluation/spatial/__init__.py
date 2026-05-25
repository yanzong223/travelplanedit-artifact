"""
Spatial Evaluation Module

Provides spatial analysis functions for travel plan evaluation,
including route centeredness, spatial clustering, and other
geometric evaluation metrics.
"""

from .route_centering import compute_route_centeredness

__all__ = [
    "compute_route_centeredness"
]