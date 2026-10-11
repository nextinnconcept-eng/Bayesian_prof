"""Shared CLI plumbing for the closed-loop scripts."""

from __future__ import annotations

import argparse

from dssc_discovery.domain.enums import SpeciesCode


def species_arg(value: str) -> SpeciesCode:
    for species in SpeciesCode:
        if value in (species.value, species.name):
            return species
    raise argparse.ArgumentTypeError(
        f"unknown species {value!r}; choose from {[s.name for s in SpeciesCode]}"
    )
