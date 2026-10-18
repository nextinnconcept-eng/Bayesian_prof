"""Plain-language guide to every value the owner enters (or uploads).

Written for someone with no machine-learning background: every field has
a one-sentence explanation of what it is, its unit, a realistic example,
and (where meaningful) the typical range the optimizer explores. The same
specs drive the manual form, the downloadable CSV template and the error
messages, so the wording is defined once.

Columns are the wide-format names the row-ingestion engine already uses
(ingestion/manual_entry.py); nothing here invents a second format.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from enum import StrEnum

from dssc_discovery.domain.enums import (
    CounterElectrodeType,
    ElectrolyteType,
    IlluminationProtocol,
    SolventType,
    SpeciesCode,
    SurfaceTreatment,
)
from dssc_discovery.optimization.search_space import ContinuousParam, search_space_for

GROUP_SPECIES = "Plant"
GROUP_RECIPE = "Recipe: dye extraction"
GROUP_SENSITIZATION = "Recipe: dyeing the TiO₂ film"
GROUP_PHOTOANODE = "Recipe: building the TiO₂ film"
GROUP_CELL = "The finished solar cell"
GROUP_RESULT = "Test result (measured under light)"


@dataclass(frozen=True)
class FieldSpec:
    name: str  # CSV column name, same as the ingestion engine
    label: str
    group: str
    help: str  # one plain sentence
    example: str
    kind: str  # "choice" | "number" | "date"
    unit: str = ""
    choices: tuple[tuple[str, str], ...] = ()  # (stored value, plain label)
    required: bool = True
    optional_note: str = ""  # what happens if left blank (optional fields only)

    @property
    def is_recipe(self) -> bool:
        return self.group in (GROUP_RECIPE, GROUP_SENSITIZATION, GROUP_PHOTOANODE)


def _choices(enum_cls: type[StrEnum], labels: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple((m.value, labels.get(m.value, m.value.replace("_", " "))) for m in enum_cls)


SPECIES_CHOICES = tuple((s.value, s.value) for s in SpeciesCode)

SOLVENT_CHOICES = _choices(
    SolventType,
    {
        "acidified_ethanol": "Ethanol with a little acid added",
        "mixed": "A mixture of solvents",
        "other": "Something else",
    },
)
SURFACE_CHOICES = _choices(
    SurfaceTreatment,
    {
        "none": "No extra surface treatment",
        "ticl4": "TiCl₄ (titanium tetrachloride) treatment",
        "mgo_coating": "MgO coating",
        "other": "Something else",
    },
)
ELECTROLYTE_CHOICES = _choices(
    ElectrolyteType,
    {"iodide_triiodide": "Iodide / triiodide (the common one)", "cobalt_complex": "Cobalt complex"},
)
COUNTER_CHOICES = _choices(
    CounterElectrodeType,
    {"platinum": "Platinum", "carbon": "Carbon", "pedot_pss": "PEDOT:PSS"},
)
ILLUMINATION_CHOICES = _choices(
    IlluminationProtocol,
    {
        "am1.5g_1_sun": "Standard sunlight simulator, full power (100 mW/cm²)",
        "am1.5g_half_sun": "Standard sunlight simulator, half power (50 mW/cm²)",
        "dark": "In the dark",
        "other": "Something else",
    },
)

_RECIPE_FIELDS = (
    FieldSpec(
        "solvent",
        "Extraction solvent",
        GROUP_RECIPE,
        "The liquid used to pull the dye out of the plant material.",
        "ethanol",
        "choice",
        choices=SOLVENT_CHOICES,
    ),
    FieldSpec(
        "solvent_ratio",
        "Share of the main solvent",
        GROUP_RECIPE,
        "How much of the solvent mixture is the main solvent: 1.0 means pure, "
        "0.7 means 70% main solvent and 30% water.",
        "0.7",
        "number",
        unit="fraction, 0 to 1",
    ),
    FieldSpec(
        "ph",
        "pH of the dye extract",
        GROUP_RECIPE,
        "How acidic the dye solution is. Lower than 7 is acidic, higher is basic.",
        "5.5",
        "number",
        unit="pH, 0 to 14",
    ),
    FieldSpec(
        "extraction_temperature_c",
        "Extraction temperature",
        GROUP_RECIPE,
        "Temperature while the dye was being pulled out of the plant.",
        "25",
        "number",
        unit="°C",
    ),
    FieldSpec(
        "extraction_time_min",
        "Extraction time",
        GROUP_RECIPE,
        "How long the plant material soaked or was heated in the solvent.",
        "60",
        "number",
        unit="minutes",
    ),
    FieldSpec(
        "pigment_concentration_mm",
        "Dye concentration of the extract",
        GROUP_RECIPE,
        "How strong the dye solution is (millimolar, mM). Higher means more dye.",
        "10",
        "number",
        unit="mM",
    ),
    FieldSpec(
        "adsorption_time_min",
        "Soaking time of the film in the dye",
        GROUP_SENSITIZATION,
        "How long the TiO₂ film sat in the dye so the dye could stick to it.",
        "120",
        "number",
        unit="minutes",
    ),
    FieldSpec(
        "sensitization_concentration_mm",
        "Dye concentration of the soaking bath",
        GROUP_SENSITIZATION,
        "Strength of the dye bath the film was soaked in (mM). Often the same as the "
        "extract, or a diluted version of it.",
        "0.5",
        "number",
        unit="mM",
    ),
    FieldSpec(
        "sensitization_temperature_c",
        "Soaking temperature",
        GROUP_SENSITIZATION,
        "Temperature of the dye bath while the film soaked.",
        "25",
        "number",
        unit="°C",
    ),
    FieldSpec(
        "tio2_thickness_um",
        "TiO₂ film thickness",
        GROUP_PHOTOANODE,
        "Thickness of the white titanium dioxide layer that holds the dye.",
        "12",
        "number",
        unit="micrometres (µm)",
    ),
    FieldSpec(
        "sintering_temperature_c",
        "Baking (sintering) temperature",
        GROUP_PHOTOANODE,
        "Oven temperature used to bake the TiO₂ film after coating it.",
        "450",
        "number",
        unit="°C",
    ),
    FieldSpec(
        "sintering_time_min",
        "Baking (sintering) time",
        GROUP_PHOTOANODE,
        "How long the film was baked at that temperature.",
        "30",
        "number",
        unit="minutes",
    ),
    FieldSpec(
        "surface_treatment",
        "Surface treatment of the film",
        GROUP_PHOTOANODE,
        "An optional extra coating or dip applied to the film before dyeing. "
        "Pick 'No extra surface treatment' if you did nothing special.",
        "none",
        "choice",
        choices=SURFACE_CHOICES,
    ),
)

_CELL_FIELDS = (
    FieldSpec(
        "active_area_cm2",
        "Lit area of the cell",
        GROUP_CELL,
        "The area of the cell that is exposed to light. A 0.5 cm × 0.5 cm square is 0.25.",
        "0.25",
        "number",
        unit="cm²",
    ),
    FieldSpec(
        "electrolyte_type",
        "Electrolyte",
        GROUP_CELL,
        "The liquid between the two electrodes. Choose iodide / triiodide if unsure "
        "and you used a standard iodine-based one.",
        "iodide_triiodide",
        "choice",
        choices=ELECTROLYTE_CHOICES,
    ),
    FieldSpec(
        "counter_electrode_type",
        "Counter electrode",
        GROUP_CELL,
        "The back electrode of the cell (the side opposite the dyed film).",
        "platinum",
        "choice",
        choices=COUNTER_CHOICES,
    ),
)

_RESULT_FIELDS = (
    FieldSpec(
        "illumination_protocol",
        "Light used for the test",
        GROUP_RESULT,
        "The light the cell was under when you measured it.",
        "am1.5g_1_sun",
        "choice",
        choices=ILLUMINATION_CHOICES,
    ),
    FieldSpec(
        "measurement_date",
        "Date of the test",
        GROUP_RESULT,
        "The day you measured the cell, written year-month-day.",
        "2026-10-18",
        "date",
        unit="YYYY-MM-DD",
    ),
    FieldSpec(
        "open_circuit_voltage_v",
        "Open-circuit voltage (Voc)",
        GROUP_RESULT,
        "The voltage the cell gives with nothing connected. Read from your J-V test.",
        "0.62",
        "number",
        unit="volts",
    ),
    FieldSpec(
        "short_circuit_current_density_ma_cm2",
        "Short-circuit current density (Jsc)",
        GROUP_RESULT,
        "The current per square centimetre when the cell is shorted. Read from your J-V test.",
        "8.5",
        "number",
        unit="mA/cm²",
    ),
    FieldSpec(
        "fill_factor",
        "Fill factor (FF)",
        GROUP_RESULT,
        "How 'square' your J-V curve is, as a fraction between 0 and 1. "
        "If your instrument shows a percentage such as 62%, enter 0.62.",
        "0.62",
        "number",
        unit="fraction, 0 to 1",
    ),
    FieldSpec(
        "power_conversion_efficiency_percent",
        "Efficiency (PCE)",
        GROUP_RESULT,
        "How much of the light energy became electricity, in percent. "
        "The goal is to make this as high as possible.",
        "3.3",
        "number",
        unit="%",
        required=False,
        optional_note="Left blank, it is calculated from Voc × Jsc × FF ÷ light power.",
    ),
)

SPECIES_FIELD = FieldSpec(
    "species_code",
    "Plant",
    GROUP_SPECIES,
    "Which plant the dye came from.",
    SpeciesCode.BETA_VULGARIS.value,
    "choice",
    choices=SPECIES_CHOICES,
)

# Optional columns the engine needs but a beginner should never have to invent.
_OPTIONAL_FIELDS = (
    FieldSpec(
        "condition_code",
        "Experiment name",
        GROUP_SPECIES,
        "A short name for this recipe, e.g. BV-01. Rows with the same name are treated "
        "as repeat cells of the same recipe.",
        "BV-01",
        "text",
        required=False,
        optional_note="Left blank, a name is created from the recipe.",
    ),
    FieldSpec(
        "device_code",
        "Cell name",
        GROUP_CELL,
        "A label for this individual cell, e.g. BV-01-R1.",
        "BV-01-R1",
        "text",
        required=False,
        optional_note="Left blank, a name is created automatically.",
    ),
    FieldSpec(
        "replicate_number",
        "Repeat number",
        GROUP_CELL,
        "If you built several identical cells of one recipe, number them 1, 2, 3.",
        "1",
        "number",
        required=False,
        optional_note="Left blank, repeats are numbered automatically.",
    ),
    FieldSpec(
        "instrument",
        "Test instrument",
        GROUP_RESULT,
        "The machine used for the J-V test, e.g. Keithley 2400.",
        "Keithley 2400",
        "text",
        required=False,
        optional_note="Left blank, it is recorded as 'not recorded'.",
    ),
)

ALL_FIELDS: tuple[FieldSpec, ...] = (
    SPECIES_FIELD,
    *_RECIPE_FIELDS,
    *_CELL_FIELDS,
    *_RESULT_FIELDS,
    *_OPTIONAL_FIELDS,
)
FIELDS_BY_NAME: dict[str, FieldSpec] = {f.name: f for f in ALL_FIELDS}
RECIPE_FIELD_NAMES: tuple[str, ...] = tuple(f.name for f in _RECIPE_FIELDS)
REQUIRED_FIELD_NAMES: tuple[str, ...] = tuple(f.name for f in ALL_FIELDS if f.required)

# Columns, in the order the template shows them.
TEMPLATE_COLUMNS: tuple[str, ...] = tuple(f.name for f in ALL_FIELDS)

# Columns of a lab sheet: recipe filled in, everything measured left blank.
LAB_SHEET_RESULT_COLUMNS: tuple[str, ...] = tuple(
    f.name
    for f in (*_CELL_FIELDS, *_RESULT_FIELDS, *_OPTIONAL_FIELDS)
    if f.name != "condition_code"
)


def plain_value(spec: FieldSpec, stored: str) -> str:
    """The plain-language label for a stored choice value (or the value itself)."""
    for value, label in spec.choices:
        if value == stored:
            return label
    return stored


def typical_range(species: SpeciesCode, name: str) -> tuple[float, float] | None:
    """The range the optimizer explores for a recipe field, as a hint only."""
    if name not in RECIPE_FIELD_NAMES:
        return None
    param = search_space_for(species).param(name)
    if isinstance(param, ContinuousParam):
        return (param.low, param.high)
    return None


def example_row() -> dict[str, str]:
    return {f.name: f.example for f in ALL_FIELDS}


def template_csv() -> str:
    """A CSV with a header and one fully worked example row."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(TEMPLATE_COLUMNS), lineterminator="\n")
    writer.writeheader()
    writer.writerow(example_row())
    return buffer.getvalue()


def instructions_text() -> str:
    """Written instructions that go with the template, for a first-time user."""
    lines = [
        "HOW TO FILL IN THE TEMPLATE",
        "",
        "One row = one finished solar cell that you built AND tested.",
        "If you built three identical cells of the same recipe, write three rows with the",
        "same recipe values (and the same experiment name, or leave the name blank).",
        "",
        "Use the units shown below. Do not leave a required column empty.",
        "",
    ]
    group = ""
    for spec in ALL_FIELDS:
        if spec.group != group:
            group = spec.group
            lines += ["", group.upper()]
        need = "required" if spec.required else "optional"
        unit = f" [{spec.unit}]" if spec.unit else ""
        lines.append(f"- {spec.name} ({need}){unit}: {spec.help}")
        if spec.choices:
            options = ", ".join(value for value, _ in spec.choices)
            lines.append(f"    choose one of: {options}")
        if spec.optional_note:
            lines.append(f"    {spec.optional_note}")
        lines.append(f"    example: {spec.example}")
    return "\n".join(lines) + "\n"
