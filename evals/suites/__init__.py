"""Specific suites, not a generic framework.

RAGAS, DeepEval, promptfoo, and Langfuse own the generic-eval-framework space and own
it well. The differentiation here is knowing which failures are worth testing for —
cross-tenant leakage under competing documents, wrong-and-confident rate, and ablation
outcome drift — never the runner underneath them.
"""
