"""Feature registry: the single source of truth for which columns are
candidate ML features, their type, their source, and the sample-size
feature-count budget that constrains how many of them any model may use
at once. No SQLAlchemy import here (Section D): this is ML-facing code,
DB-agnostic by construction -- source_table/source_field are plain
strings the database layer's own tests verify against, not a live
binding to it.

Two distinct feature STAGES -- not interchangeable:

- "design": available before fabrication (species + the
  ExperimentalConditionVector's 13 parameters, plus fixed device
  protocol choices). These are the ONLY features the BO surrogate
  (Chunk 6.x) may ever use: BO predicts expected PCE for a condition
  that doesn't physically exist yet, so a characterization-stage
  feature isn't just methodologically awkward there, it's literally
  unavailable at recommendation time. Using one anyway would silently
  produce a model that can't be deployed for its actual purpose.
- "characterization": available only after a post-fabrication
  chemical/optical/electrochemical measurement (UV-Vis lambda_max, EIS
  Rct/Rrec, ...). Legitimate for the explanatory/SHAP model (Chunks
  4.x-5.x, Papers 2-3) -- "does lambda_max correlate with PCE" is real,
  useful science -- but must never reach the BO surrogate.

FEATURE_BUDGET (PLAN.txt Section H): at N~70-85 experimental
conditions, a defensible ceiling is roughly N/10 features for tree
ensembles (~7-8) and N/5 for regularized linear/GP models (~14).
Encoded here as constants so "just use all the columns" can't happen by
accident -- model-building code (evaluation/, once it exists) imports
these rather than picking its own number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FeatureStage = Literal["design", "characterization"]
FeatureDType = Literal["continuous", "categorical"]

FEATURE_BUDGET: dict[str, int] = {
    "tree": 8,
    "linear_or_gp": 14,
}


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    dtype: FeatureDType
    stage: FeatureStage
    source_table: str
    source_field: str
    description: str


FEATURE_REGISTRY: tuple[FeatureSpec, ...] = (
    # --- design-time: species ---------------------------------------------
    FeatureSpec(
        "species_code",
        "categorical",
        "design",
        "experimental_conditions",
        "species_code",
        "Plant species",
    ),
    # --- design-time: ExperimentalConditionVector (Chunk 1.2) -------------
    FeatureSpec(
        "solvent",
        "categorical",
        "design",
        "experimental_conditions",
        "solvent",
        "Extraction solvent",
    ),
    FeatureSpec(
        "solvent_ratio",
        "continuous",
        "design",
        "experimental_conditions",
        "solvent_ratio",
        "Primary solvent volume fraction",
    ),
    FeatureSpec("ph", "continuous", "design", "experimental_conditions", "ph", "Extraction pH"),
    FeatureSpec(
        "extraction_temperature_c",
        "continuous",
        "design",
        "experimental_conditions",
        "extraction_temperature_c",
        "Extraction temperature",
    ),
    FeatureSpec(
        "extraction_time_min",
        "continuous",
        "design",
        "experimental_conditions",
        "extraction_time_min",
        "Extraction time",
    ),
    FeatureSpec(
        "pigment_concentration_mm",
        "continuous",
        "design",
        "experimental_conditions",
        "pigment_concentration_mm",
        "Target pigment concentration",
    ),
    FeatureSpec(
        "adsorption_time_min",
        "continuous",
        "design",
        "experimental_conditions",
        "adsorption_time_min",
        "Dye adsorption time",
    ),
    FeatureSpec(
        "sensitization_concentration_mm",
        "continuous",
        "design",
        "experimental_conditions",
        "sensitization_concentration_mm",
        "Sensitization bath concentration",
    ),
    FeatureSpec(
        "sensitization_temperature_c",
        "continuous",
        "design",
        "experimental_conditions",
        "sensitization_temperature_c",
        "Sensitization temperature",
    ),
    FeatureSpec(
        "tio2_thickness_um",
        "continuous",
        "design",
        "experimental_conditions",
        "tio2_thickness_um",
        "TiO2 film thickness",
    ),
    FeatureSpec(
        "sintering_temperature_c",
        "continuous",
        "design",
        "experimental_conditions",
        "sintering_temperature_c",
        "Sintering temperature",
    ),
    FeatureSpec(
        "sintering_time_min",
        "continuous",
        "design",
        "experimental_conditions",
        "sintering_time_min",
        "Sintering time",
    ),
    FeatureSpec(
        "surface_treatment",
        "categorical",
        "design",
        "experimental_conditions",
        "surface_treatment",
        "Photoanode surface treatment",
    ),
    # --- design-time: Device (fixed protocol choices at fabrication, known
    # before any J-V/EIS characterization happens) -------------------------
    FeatureSpec(
        "active_area_cm2",
        "continuous",
        "design",
        "devices",
        "active_area_cm2",
        "Device active area",
    ),
    FeatureSpec(
        "electrolyte_type",
        "categorical",
        "design",
        "devices",
        "electrolyte_type",
        "Electrolyte formulation",
    ),
    FeatureSpec(
        "counter_electrode_type",
        "categorical",
        "design",
        "devices",
        "counter_electrode_type",
        "Counter electrode material",
    ),
    # --- characterization-stage: extraction outcome ------------------------
    FeatureSpec(
        "extraction_yield_percent",
        "continuous",
        "characterization",
        "pigment_extract_batches",
        "extraction_yield_percent",
        "Measured extraction yield",
    ),
    # --- characterization-stage: UV-Vis derived descriptors (Chunk 2.2) ---
    # sourced from the generic derived_descriptors.value column, filtered
    # by descriptor_name == this feature's name (Chunk 2.2's naming
    # convention already matches these exactly).
    FeatureSpec(
        "lambda_max_nm",
        "continuous",
        "characterization",
        "derived_descriptors",
        "value",
        "UV-Vis absorption peak wavelength",
    ),
    FeatureSpec(
        "bandwidth_nm",
        "continuous",
        "characterization",
        "derived_descriptors",
        "value",
        "UV-Vis absorption peak FWHM",
    ),
    FeatureSpec(
        "integrated_absorption",
        "continuous",
        "characterization",
        "derived_descriptors",
        "value",
        "UV-Vis integrated absorption",
    ),
    # --- characterization-stage: EIS (Chunk 2.4) ---------------------------
    FeatureSpec(
        "series_resistance_ohm",
        "continuous",
        "characterization",
        "electrochemical_measurements",
        "series_resistance_ohm",
        "EIS series resistance",
    ),
    FeatureSpec(
        "charge_transfer_resistance_ohm",
        "continuous",
        "characterization",
        "electrochemical_measurements",
        "charge_transfer_resistance_ohm",
        "EIS charge-transfer resistance",
    ),
    FeatureSpec(
        "recombination_resistance_ohm",
        "continuous",
        "characterization",
        "electrochemical_measurements",
        "recombination_resistance_ohm",
        "EIS recombination resistance",
    ),
    FeatureSpec(
        "electron_lifetime_ms",
        "continuous",
        "characterization",
        "electrochemical_measurements",
        "electron_lifetime_ms",
        "EIS-derived electron lifetime",
    ),
)


def design_features() -> tuple[FeatureSpec, ...]:
    """Features usable by the BO surrogate (Chunk 6.x) -- available before fabrication."""
    return tuple(f for f in FEATURE_REGISTRY if f.stage == "design")


def characterization_features() -> tuple[FeatureSpec, ...]:
    """Features usable only by the explanatory/SHAP model (Chunks 4.x-5.x)."""
    return tuple(f for f in FEATURE_REGISTRY if f.stage == "characterization")


def feature_names(stage: FeatureStage | None = None) -> tuple[str, ...]:
    if stage is None:
        return tuple(f.name for f in FEATURE_REGISTRY)
    return tuple(f.name for f in FEATURE_REGISTRY if f.stage == stage)


def get_feature(name: str) -> FeatureSpec:
    for feature in FEATURE_REGISTRY:
        if feature.name == name:
            return feature
    raise KeyError(f"{name!r} is not a registered candidate feature")
