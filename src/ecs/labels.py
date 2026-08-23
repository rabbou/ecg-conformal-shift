"""Building one comparable myocardial-infarction label on two corpora.

The two corpora do not speak the same language.  PTB-XL annotates with SCP-ECG
statements carrying a 0-100 likelihood; Shandong annotates with AHA codes
carrying an acute / recent / old modifier.  Everything that makes those two
comparable -- or fails to -- is decided here and nowhere else.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

import pandas as pd

from .config import (
    PTBXL_INJURY_STATEMENTS,
    SPH_MI_CODES,
    SPH_MODIFIER_OLD,
)

__all__ = ["MILabelSpec", "ptbxl_mi_label", "sph_mi_label"]


@dataclass(frozen=True)
class MILabelSpec:
    """How strictly to read "this ECG shows an infarction".

    include_injury
        Keep PTB-XL's subendocardial-injury statements inside the positive
        class.  They ship inside the MI superclass but describe ischaemic
        injury rather than an infarct pattern.
    min_likelihood
        Drop PTB-XL statements annotated below this confidence.  PTB-XL records
        a 0-100 likelihood per statement and a large minority of MI statements
        sit at 15.
    chronic_only
        Restrict both corpora to old / chronic infarction.  This is the one
        setting under which the two label sets plausibly denote the same thing,
        at the cost of most of the positive cases.
    """

    include_injury: bool = True
    min_likelihood: float = 0.0
    chronic_only: bool = False


def ptbxl_mi_label(
    database: pd.DataFrame,
    statements: pd.DataFrame,
    spec: MILabelSpec | None = None,
) -> pd.Series:
    """Boolean MI label for every PTB-XL record, indexed like ``database``.

    ``database`` is ptbxl_database.csv, ``statements`` is scp_statements.csv.
    """
    spec = spec or MILabelSpec()
    diagnostic = statements[statements["diagnostic"] == 1]
    mi_statements = set(diagnostic.index[diagnostic["diagnostic_class"] == "MI"])
    if not spec.include_injury:
        mi_statements -= set(PTBXL_INJURY_STATEMENTS)
    if spec.chronic_only:
        # Subendocardial injury is by definition the acute picture; what remains
        # are the infarct-pattern statements, which PTB-XL does not date.
        mi_statements -= set(PTBXL_INJURY_STATEMENTS)

    codes = database["scp_codes"].apply(_parse_scp_codes)

    def is_mi(entry: dict[str, float]) -> bool:
        return any(
            statement in mi_statements and likelihood >= spec.min_likelihood
            for statement, likelihood in entry.items()
        )

    return codes.apply(is_mi).rename("mi")


def sph_mi_label(metadata: pd.DataFrame, spec: MILabelSpec | None = None) -> pd.Series:
    """Boolean MI label for every Shandong record, indexed like ``metadata``.

    ``metadata`` is the corpus metadata.csv, whose ``AHA_Code`` column holds
    semicolon-separated codes, each optionally suffixed ``+<modifier>``.
    """
    spec = spec or MILabelSpec()
    parsed = metadata["AHA_Code"].astype(str).apply(_parse_aha_codes)

    def is_mi(entries: list[tuple[str, str | None]]) -> bool:
        hits = [(code, modifier) for code, modifier in entries if code in SPH_MI_CODES]
        if not hits:
            return False
        if spec.chronic_only:
            return any(modifier == SPH_MODIFIER_OLD for _, modifier in hits)
        return True

    return parsed.apply(is_mi).rename("mi")


def _parse_scp_codes(raw: str | dict[str, float]) -> dict[str, float]:
    if isinstance(raw, dict):
        return raw
    parsed = ast.literal_eval(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"scp_codes did not parse to a dict: {raw!r}")
    return parsed


def _parse_aha_codes(raw: str) -> list[tuple[str, str | None]]:
    out: list[tuple[str, str | None]] = []
    for token in raw.split(";"):
        token = token.strip()
        if not token:
            continue
        code, _, modifier = token.partition("+")
        out.append((code, modifier or None))
    return out
