"""
Synthetic data generation package for the Health Monitor project.

Provides realistic synthetic wearable sensor data for training and
testing health-condition detection models before real sensor data
becomes available.
"""

from data.synthetic.generator import SyntheticDataGenerator

__all__ = ["SyntheticDataGenerator"]
