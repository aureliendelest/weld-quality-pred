"""Specifications: the thresholds that make a weld conforming, per class of consumable or per use.

A weld is called qualitative when its all-weld-metal properties meet a specification (doc/06_quality_decision_plan.md).
A specification is a set of requirements on the targets of the project, taken from a classification standard or an
application code. It is used in two ways:
- on measured values, to build the conformity label of the dataset (`conformity`), with an explicit "unknown";
- on predicted intervals, to decide for a new weld (`decide`): conforming, non conforming, to test or to measure.

Every specification of `SPECIFICATIONS` records where its numbers come from and how well they were checked.
"""
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

CONFORMING, NON_CONFORMING, UNKNOWN = "conforming", "non_conforming", "unknown"
TO_TEST, TO_MEASURE = "to_test", "to_measure"
STATUSES = ("verified", "secondary", "illustrative")     # see the table of doc/06
CHARPY, CHARPY_TEMPERATURE = "charpy_toughness_j", "charpy_temp_c"


@dataclass(frozen=True)
class Requirement:
    target: str
    minimum: float | None = None
    maximum: float | None = None
    test_temperature_c: float | None = None     # the temperature at which the Charpy energy is required

    def holds(self, value) -> bool:
        return (self.minimum is None or value >= self.minimum) and (self.maximum is None or value <= self.maximum)


@dataclass(frozen=True)
class Decision:
    overall: str                    # conforming / non_conforming / to_test
    by_target: dict                 # target -> conforming / non_conforming / to_test / to_measure


@dataclass(frozen=True)
class Specification:
    name: str
    requirements: tuple
    use: str                        # what the class is meant for, in words
    source: str                     # where the numbers come from
    status: str                     # one of STATUSES

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}, got {self.status!r}")

    @property
    def targets(self) -> list:
        return [r.target for r in self.requirements]

    def conditions(self) -> dict:
        """Values of the test-condition features at which a prediction must be made for this specification."""
        return {CHARPY_TEMPERATURE: r.test_temperature_c for r in self.requirements
                if r.test_temperature_c is not None}

    def _measured_state(self, requirement: Requirement, df: pd.DataFrame) -> pd.Series:
        """conforming / non_conforming / unknown of each row for one requirement.

        A Charpy energy falls as the test temperature falls, so a result at or below the required temperature
        proves conformity when it passes, and a result at or above it proves non conformity when it fails; the
        other two cases say nothing.
        """
        value = df[requirement.target]
        met = pd.Series(True, index=df.index)
        if requirement.minimum is not None:
            met &= value >= requirement.minimum
        if requirement.maximum is not None:
            met &= value <= requirement.maximum
        known = value.notna()
        proven, refuted = known & met, known & ~met
        if requirement.test_temperature_c is not None:
            temperature = df[CHARPY_TEMPERATURE]
            proven &= temperature <= requirement.test_temperature_c
            refuted &= temperature >= requirement.test_temperature_c
        state = pd.Series(UNKNOWN, index=df.index)
        state[proven], state[refuted] = CONFORMING, NON_CONFORMING
        return state

    def states(self, df: pd.DataFrame) -> pd.DataFrame:
        """One column per required target: the state of each measured weld for that requirement."""
        return pd.DataFrame({r.target: self._measured_state(r, df) for r in self.requirements})

    def conformity(self, df: pd.DataFrame) -> pd.Series:
        """Conforming when every requirement is proven, non conforming when one is failed, unknown otherwise."""
        states = self.states(df)
        overall = pd.Series(UNKNOWN, index=df.index)
        overall[(states == CONFORMING).all(axis=1)] = CONFORMING
        overall[(states == NON_CONFORMING).any(axis=1)] = NON_CONFORMING
        return overall

    def decide(self, intervals: Mapping[str, tuple]) -> Decision:
        """Decision for one new weld from prediction intervals {target: (lower, upper)}.

        A requirement is met when the whole interval is on the right side of the threshold, failed when the
        whole interval is on the wrong side, and to test in between. A target without an interval (for example
        the elongation, which no model predicts on a new study) has to be measured. One failed requirement makes
        the weld non conforming; it is conforming only when every requirement is met.
        """
        by_target = {}
        for requirement in self.requirements:
            interval = intervals.get(requirement.target)
            if interval is None or any(pd.isna(bound) for bound in interval):
                by_target[requirement.target] = TO_MEASURE
                continue
            lower, upper = interval
            if requirement.holds(lower) and requirement.holds(upper):
                by_target[requirement.target] = CONFORMING
            elif (requirement.minimum is not None and upper < requirement.minimum) or \
                 (requirement.maximum is not None and lower > requirement.maximum):
                by_target[requirement.target] = NON_CONFORMING
            else:
                by_target[requirement.target] = TO_TEST
        states = set(by_target.values())
        overall = NON_CONFORMING if NON_CONFORMING in states else CONFORMING if states == {CONFORMING} else TO_TEST
        return Decision(overall, by_target)


def _aws_a51(name, yield_mpa, tensile_mpa, elongation, use, status, source, charpy=None) -> Specification:
    """AWS A5.1 class from its tension requirements, in MPa as the manufacturers print them (58 ksi = 400, 70 ksi = 480)."""
    requirements = [Requirement("yield_strength_mpa", minimum=yield_mpa),
                    Requirement("uts_mpa", minimum=tensile_mpa),
                    Requirement("elongation_pct", minimum=elongation)]
    if charpy:
        requirements.append(Requirement(CHARPY, minimum=charpy[0], test_temperature_c=charpy[1]))
    return Specification(name, tuple(requirements), use, source, status)


_LINCOLN = "https://www.lincolnelectric.com/fr-CA/Products/millenniumarc70181_smaw"
_ESAB = "https://www.weldingsuppliesfromioc.com/cdn/shop/files/Sureweld_7018_Specs.pdf"
_AWS_TABLE = "https://www.scribd.com/document/831341815/AWS-Classifications"      # secondary copy of the A5.1 tension table
_ELGA = "https://elga.evostudio.pl/wp-content/uploads/2021/07/ELGA-Classification_EN-ISO-2560-A-1.pdf"

SPECIFICATIONS = {spec.name: spec for spec in [
    Specification(
        "ISO 2560-A E42 4",
        (Requirement("yield_strength_mpa", minimum=420), Requirement("uts_mpa", minimum=500, maximum=640),
         Requirement("elongation_pct", minimum=20), Requirement(CHARPY, minimum=47, test_temperature_c=-40)),
        "yield-strength system of ISO 2560 for non-alloy and fine-grain steels, 420 MPa class, 47 J at -40 C",
        f"ISO 2560-A, as reproduced by the ELGA classification guide ({_ELGA}); not read in the standard itself",
        "secondary"),
    _aws_a51("AWS A5.1 E7018", 400, 480, 22, "mild-steel electrode of the 70 ksi class (structural work)",
             "verified", f"{_LINCOLN}; {_ESAB}; tension table {_AWS_TABLE}"),
    _aws_a51("AWS A5.1 E7018-1", 400, 480, 22, "same class with a low-temperature impact requirement",
             "verified", f"{_LINCOLN}; {_ESAB}", charpy=(27, -45)),
    _aws_a51("AWS A5.1 E6010", 331, 414, 22, "mild-steel electrode of the 60 ksi class (root passes, pipe)",
             "secondary", f"tension table of AWS A5.1 as reproduced in {_AWS_TABLE}"),
    _aws_a51("AWS A5.1 E6013", 331, 414, 17, "mild-steel electrode of the 60 ksi class, lower elongation",
             "secondary", f"tension table of AWS A5.1 as reproduced in {_AWS_TABLE}"),
]}


def get_specification(name: str) -> Specification:
    if name not in SPECIFICATIONS:
        raise KeyError(f"unknown specification {name!r}; known: {sorted(SPECIFICATIONS)}")
    return SPECIFICATIONS[name]
