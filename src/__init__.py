"""
Smart Health Tracker Insights
A comprehensive ML/DL project for health analytics
"""

__version__ = "1.0.0"
__author__ = "Ishika Dubey"

from . import preprocessing
from . import eda
from . import classification_models
from . import deep_learning_models

__all__ = [
    'preprocessing',
    'eda',
    'classification_models',
    'deep_learning_models'
]
