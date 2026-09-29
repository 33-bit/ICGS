"""Source-neutral Stage-1 executed-episode records, labels and indexed views.

Nothing in this package imports a simulator. Environment adapters (for example
``icgs.environments.robohiman``) produce records; training code consumes views.
"""

from icgs.data.stage1.schema import (
    OUTCOME_STATUSES,
    PERTURBATION_FAMILIES,
    RECOVERABILITY,
    SCHEMA_VERSION,
    validate_manifest,
)

__all__ = [
    "OUTCOME_STATUSES",
    "PERTURBATION_FAMILIES",
    "RECOVERABILITY",
    "SCHEMA_VERSION",
    "validate_manifest",
]
