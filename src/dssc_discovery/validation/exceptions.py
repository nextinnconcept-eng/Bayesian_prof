"""Exception types for cross-entity business-rule validation.

Deliberately distinct from pydantic.ValidationError, which only catches
single-model, field-level violations. Domain models (Chunks 1.1-1.3)
already reject an out-of-range value on their own via Field()/
@field_validator; these catch violations that need more than one field,
or more than one record, to detect.
"""

from __future__ import annotations


class DomainValidationError(Exception):
    """Base for all business-rule validation failures in this package."""


class DuplicateConditionError(DomainValidationError):
    """A semantically-identical ExperimentalCondition (same species and
    parameter vector, not necessarily the same condition_code) already
    exists.
    """


class PhysicallyImplausibleError(DomainValidationError):
    """A record is internally inconsistent in a way no single field's
    bounds alone could catch -- e.g. PCE not matching Voc*Jsc*FF, or
    stability retention_percent not matching the baseline/timepoint PCE
    ratio.
    """


class MissingProvenanceError(DomainValidationError):
    """A record has a missing (None) numeric result field but no
    missing_value_reason explaining why -- silent gaps are not allowed.
    """
