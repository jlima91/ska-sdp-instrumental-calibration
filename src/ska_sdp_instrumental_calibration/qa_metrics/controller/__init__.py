"""
Controller of the QA report: builds the sections of each field from the
run metadata, through :mod:`..service` and :mod:`..domain`, and renders
them with :mod:`..view`.
"""

from .report import generate_qa_metric

__all__ = ["generate_qa_metric"]
