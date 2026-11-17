"""Week 2 statistical evaluation infrastructure.

This package never redefines a quantity already frozen in ``protocol_config.json``,
``PLAN_Final.md`` §3, or the reference implementations under ``code/``. It
consumes their outputs (via ``WEEK2_ARTIFACT_CONTRACT.md``-shaped artifacts)
and adds aggregation, baselines, censoring bookkeeping, calibration, schema
validation, and the decision gate on top.

No exact-oracle quantity (``P``, ``R``, ``V*``, ``Q*``, exact normalized
return, exact ``k_dagger``, family labels) may be read by anything under
``evaluation.calibration`` or ``evaluation.baselines.structural_only`` as a
predictive feature. See ``evaluation.validation.leakage``.
"""

from __future__ import annotations
