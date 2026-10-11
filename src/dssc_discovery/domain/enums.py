"""Controlled vocabularies for every categorical domain field.

Every enum here is a StrEnum: members compare equal to their string value,
which round-trips cleanly through Pydantic (JSON/dict) and, later, through
SQLAlchemy Enum columns -- no separate serializer needed.

Where lab reality can plausibly fall outside a fixed list (solvent choice,
surface treatment, ...), an OTHER member is included as an escape hatch,
paired with a free-text `notes` field on the owning domain model (added in
the Chunk 1.x domain models) so nothing gets silently miscategorized.
Fields with no legitimate "other" case (SpeciesCode, ClosedLoopState,
DesignSource) intentionally have none -- an unmatched value there should
fail loudly, not get swallowed into a catch-all.
"""

from __future__ import annotations

from enum import StrEnum


class SpeciesCode(StrEnum):
    """The four plant species in scope for this research programme. This is
    a closed set by design (see PLAN.txt Section 1) -- adding a fifth
    species is a deliberate scope change, not routine data entry, so there
    is no OTHER member here."""

    TELFAIRIA_OCCIDENTALIS = "Telfairia occidentalis"
    THUNBERGIA_ERECTA = "Thunbergia erecta"
    BETA_VULGARIS = "Beta vulgaris"
    CHROMOLAENA_ODORATA = "Chromolaena odorata"


class PlantPart(StrEnum):
    LEAF = "leaf"
    FLOWER = "flower"
    FRUIT = "fruit"
    ROOT = "root"
    STEM = "stem"
    BRACT = "bract"
    WHOLE_PLANT = "whole_plant"
    OTHER = "other"


class PigmentClass(StrEnum):
    ANTHOCYANIN = "anthocyanin"
    BETALAIN = "betalain"
    CHLOROPHYLL = "chlorophyll"
    CAROTENOID = "carotenoid"
    FLAVONOID = "flavonoid"
    OTHER = "other"


class SolventType(StrEnum):
    WATER = "water"
    ETHANOL = "ethanol"
    METHANOL = "methanol"
    ACETONE = "acetone"
    ACIDIFIED_ETHANOL = "acidified_ethanol"
    MIXED = "mixed"
    OTHER = "other"


class SurfaceTreatment(StrEnum):
    NONE = "none"
    TICL4 = "ticl4"
    MGO_COATING = "mgo_coating"
    OTHER = "other"


class CounterElectrodeType(StrEnum):
    PLATINUM = "platinum"
    CARBON = "carbon"
    PEDOT_PSS = "pedot_pss"
    OTHER = "other"


class ElectrolyteType(StrEnum):
    IODIDE_TRIIODIDE = "iodide_triiodide"
    COBALT_COMPLEX = "cobalt_complex"
    OTHER = "other"


class EquivalentCircuitModel(StrEnum):
    """EIS fit model used to extract Rs/Rct/Rrec/electron lifetime."""

    RANDLES = "randles"
    TRANSMISSION_LINE = "transmission_line"
    OTHER = "other"


class IlluminationProtocol(StrEnum):
    AM1_5G_1_SUN = "am1.5g_1_sun"
    AM1_5G_HALF_SUN = "am1.5g_half_sun"
    DARK = "dark"
    OTHER = "other"


class MeasurementType(StrEnum):
    """Raw instrument modality -- what kind of RawSpectrum/RawCurve this is."""

    UV_VIS = "uv_vis"
    FTIR = "ftir"
    HPLC = "hplc"
    LC_MS = "lc_ms"
    EIS = "eis"
    JV_CURVE = "jv_curve"


class MissingValueReason(StrEnum):
    INSTRUMENT_FAILURE = "instrument_failure"
    NOT_MEASURED = "not_measured"
    BELOW_DETECTION_LIMIT = "below_detection_limit"
    SAMPLE_LOST = "sample_lost"
    OTHER = "other"


class DesignSource(StrEnum):
    """Provenance of an ExperimentalCondition -- which stage of the DoE/BO
    programme proposed it. Used by the forward-extrapolation validation
    split (PLAN.txt Chunk 4.5, Section H strategy B), so values must match
    actual experimental provenance exactly -- no OTHER member."""

    INITIAL_DOE = "initial_doe"
    BO_CYCLE_1 = "bo_cycle_1"
    BO_CYCLE_2 = "bo_cycle_2"
    BO_CYCLE_3 = "bo_cycle_3"
    MANUAL = "manual"


class ClosedLoopState(StrEnum):
    """States from PLAN.txt Section K. No OTHER member -- an unrecognized
    state is a bug, not a data-entry edge case."""

    PROPOSED = "proposed"
    APPROVED = "approved"
    FABRICATED = "fabricated"
    MEASURED = "measured"
    VALIDATED = "validated"
    INGESTED = "ingested"
    MODEL_UPDATED = "model_updated"
    REJECTED = "rejected"
    FAILED_FABRICATION = "failed_fabrication"
    FAILED_MEASUREMENT = "failed_measurement"
    SUPERSEDED = "superseded"


class HypothesisStatus(StrEnum):
    """HypothesisLog lifecycle (PLAN.txt Section I / Chunk 5.4)."""

    OPEN = "open"
    TESTING = "testing"
    CONFIRMED = "confirmed"
    REFUTED = "refuted"
