"""How much real data the owner has, what that unlocks, and what to add next.

Levels (per plant species, counting real INGESTED/MODEL_UPDATED experiments):

    none       0 experiments           -> get a spread-out starting set to run in the lab
    too_few    1 .. early_min - 1      -> add N more experiments for a first estimate
    early      early_min .. full - 1   -> low-confidence "early estimate" allowed
    full       >= full                 -> the optimizer's normal, trusted mode

The two thresholds come from config (min_real_conditions_for_early_estimate,
min_real_conditions_for_retraining), never hard-coded here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from dssc_discovery.closed_loop.real_workflow import real_trained_condition_count
from dssc_discovery.config import get_config
from dssc_discovery.domain.enums import SpeciesCode
from dssc_discovery.features.dataset_builder import build_ml_dataset
from dssc_discovery.optimization.search_space import ContinuousParam, search_space_for
from dssc_discovery.user_data.field_guide import FIELDS_BY_NAME, RECIPE_FIELD_NAMES

NONE = "none"
TOO_FEW = "too_few"
EARLY = "early"
FULL = "full"


@dataclass
class Readiness:
    species: SpeciesCode
    level: str
    n_experiments: int
    n_devices: int
    early_min: int
    full_min: int
    has_synthetic_rows: bool
    unchanged_settings: list[str] = field(default_factory=list)

    @property
    def needed_for_early(self) -> int:
        return max(0, self.early_min - self.n_experiments)

    @property
    def needed_for_full(self) -> int:
        return max(0, self.full_min - self.n_experiments)

    @property
    def can_predict(self) -> bool:
        return self.level in (EARLY, FULL) and not self.has_synthetic_rows

    @property
    def headline(self) -> str:
        name = self.species.value
        if self.has_synthetic_rows:
            return (
                "This database contains simulated demo rows, so real predictions are "
                "switched off. Use 'My real data' mode (sidebar)."
            )
        if self.level == NONE:
            return f"No real results for {name} yet. Start with the suggested experiments."
        if self.level == TOO_FEW:
            return (
                f"{self.n_experiments} real experiment(s) for {name}. Add "
                f"{self.needed_for_early} more (different recipes) to unlock a first "
                "early estimate."
            )
        if self.level == EARLY:
            return (
                f"{self.n_experiments} real experiments for {name}: an early estimate is "
                f"available, but treat it as a rough guide. {self.needed_for_full} more "
                "experiments give the full, trusted mode."
            )
        return f"{self.n_experiments} real experiments for {name}: full mode is available."


def classify(n_experiments: int, early_min: int, full_min: int) -> str:
    if n_experiments <= 0:
        return NONE
    if n_experiments < early_min:
        return TOO_FEW
    if n_experiments < full_min:
        return EARLY
    return FULL


def assess_readiness(session: Session, species: SpeciesCode) -> Readiness:
    config = get_config()
    n_experiments = real_trained_condition_count(session, species)
    data = build_ml_dataset(session)
    rows = data[data["species_code"].astype(str) == species.value] if not data.empty else data
    has_synthetic = bool(data["is_synthetic"].any()) if not data.empty else False
    real_rows = rows[~rows["is_synthetic"]] if not rows.empty else rows
    return Readiness(
        species=species,
        level=classify(
            n_experiments,
            config.min_real_conditions_for_early_estimate,
            config.min_real_conditions_for_retraining,
        ),
        n_experiments=n_experiments,
        n_devices=len(real_rows),
        early_min=config.min_real_conditions_for_early_estimate,
        full_min=config.min_real_conditions_for_retraining,
        has_synthetic_rows=has_synthetic,
        unchanged_settings=_unchanged_settings(real_rows, species),
    )


def _unchanged_settings(rows, species: SpeciesCode) -> list[str]:
    """Recipe settings that were identical in every experiment so far: the
    model cannot learn anything about them yet. Advice only (>= 3 experiments)."""
    if rows is None or rows.empty or rows["group_id"].nunique() < 3:
        return []
    space = search_space_for(species)
    unchanged = []
    for name in RECIPE_FIELD_NAMES:
        if name not in rows.columns:
            continue
        param = space.param(name)
        if isinstance(param, ContinuousParam):
            if rows[name].astype(float).nunique() == 1:
                unchanged.append(FIELDS_BY_NAME[name].label)
        elif rows[name].astype(str).nunique() == 1:
            unchanged.append(FIELDS_BY_NAME[name].label)
    return unchanged
