"""Evaluation harness for the supervisor itself.

The unit tests answer "does each detector behave as coded?". They cannot answer
the only question that matters operationally: *does the supervisor actually
help, and at what cost?* This package builds a deterministic session generator
with known-answer faults, runs them through the real policy engine, and scores
the verdicts against ground truth (see :mod:`agent_shepherd.eval.metrics`).

Run it from CI with::

    python -m agent_shepherd.eval.harness --seed 7 --sessions 6 --steps 50
"""
