"""
Evaluation module for health symptom detection system.

Provides comprehensive metrics computation, publication-quality visualization,
and automated report generation for model evaluation.

Usage:
    from src.evaluation import EvaluationReport
    
    report = EvaluationReport(models_dir="models", output_dir="evaluation_results")
    report.generate_all(test_data)
"""

from .metrics import MetricsCalculator
from .visualizer import EvaluationVisualizer
from .report import EvaluationReport

__all__ = ["MetricsCalculator", "EvaluationVisualizer", "EvaluationReport"]
