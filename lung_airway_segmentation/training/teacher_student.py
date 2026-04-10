"""Semi-supervised training loop utilities.

This file should define the mechanics of teacher-student training, including:
- EMA teacher updates
- labeled and unlabeled batch handling
- pseudo-label generation
- uncertainty-aware consistency terms

Keep these details out of the baseline training loop.
"""
