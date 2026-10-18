"""Evaluation harness for the CSAM agent.

Separate from `app/` because nothing here runs in production, and separate
from `tests/` because the dataset and the metrics are artefacts in their own
right — they are read by people, not only by pytest.
"""
