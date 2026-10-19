"""
User-defined display order for groups and factor levels.

Display only: nothing here is used by the statistics, so changing an order
never invalidates cached results. Pure pandas/stdlib (no Streamlit import)
so the helpers can be unit-tested and reused by the app, the plotting
module and the report writer.

Conventions
-----------
* An order is a list of level values; position 1 (index 0) is furthest left.
* Levels are matched on ``str(x).strip()``, so labels such as
  ``'DDC + Ang 1-7 '`` (trailing space) match ``'DDC + Ang 1-7'``.
* Functions return the *observed* objects, never the preferred ones, so a
  plotting site keeps whatever value type it already used for subsetting.
"""

from itertools import product
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "level_key",
    "default_levels",
    "apply_order",
    "positions_to_order",
    "order_table",
    "combined_order",
]


def level_key(x) -> Optional[str]:
    """Matching key for a level: ``str(x).strip()``; ``None`` for missing values."""
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except (TypeError, ValueError):
        pass
    return str(x).strip()


def default_levels(values: Iterable, mode: str = "appearance") -> List:
    """
    Today's default ordering, without missing values.

    mode='appearance' -> order of first appearance (``unique()``)
    mode='sorted'     -> ``sorted(unique())``
    """
    seen = pd.unique(pd.Series(list(values), dtype=object))
    levels = [v for v in seen if level_key(v) is not None and level_key(v).lower() != "nan"]
    if mode == "sorted":
        try:
            return sorted(levels)
        except TypeError:
            return sorted(levels, key=lambda v: str(v))
    if mode != "appearance":
        raise ValueError(f"unknown mode: {mode!r}")
    return levels


def apply_order(observed: Sequence, preferred: Optional[Sequence]) -> List:
    """
    Reorder ``observed`` levels by ``preferred``.

    Levels named in ``preferred`` come first, in that order; observed levels
    not named keep their original relative order and follow. Preferred levels
    that are not observed (e.g. a group with no data in this panel) are
    skipped. With no preferred order, ``observed`` is returned unchanged.
    """
    observed = list(observed)
    if not preferred:
        return observed
    rank: Dict[str, int] = {}
    for i, p in enumerate(preferred):
        k = level_key(p)
        if k is not None and k not in rank:
            rank[k] = i
    ranked = [o for o in observed if level_key(o) in rank]
    ranked.sort(key=lambda o: rank[level_key(o)])          # stable
    rest = [o for o in observed if level_key(o) not in rank]
    return ranked + rest


def order_table(levels: Sequence) -> pd.DataFrame:
    """Editable Group/Position table for the UI (position 1 = furthest left)."""
    levels = list(levels)
    return pd.DataFrame({"Group": [str(v) for v in levels],
                         "Position": list(range(1, len(levels) + 1))})


def positions_to_order(table: pd.DataFrame) -> Tuple[Optional[List[str]], Optional[str]]:
    """
    Convert an edited Group/Position table into an ordered list of groups.

    Positions must be whole numbers and unique; gaps (1, 3, 7) are allowed.
    Returns (order, None) on success or (None, error message).
    """
    if table is None or len(table) == 0:
        return [], None
    if not {"Group", "Position"}.issubset(table.columns):
        return None, "Table needs 'Group' and 'Position' columns."
    pos = pd.to_numeric(table["Position"], errors="coerce")
    bad = table.loc[pos.isna() | (pos != np.floor(pos)), "Group"].tolist()
    if bad:
        return None, "Positions must be whole numbers (check: " + ", ".join(map(str, bad)) + ")."
    dup = pos[pos.duplicated(keep=False)]
    if len(dup):
        clash = sorted(set(int(p) for p in dup))
        return None, "Each position can be used once (repeated: " + ", ".join(map(str, clash)) + ")."
    ordered = table.assign(_p=pos.astype(int)).sort_values("_p", kind="stable")
    return [str(g) for g in ordered["Group"]], None


def combined_order(factor_orders: Sequence[Sequence], sep: str = " - ") -> List[str]:
    """
    Order of combined factor labels ('A - B - C'), built as the app builds
    them (``astype(str)`` joined by ``sep``): factor A varies slowest.
    """
    if not factor_orders:
        return []
    return [sep.join(str(level) for level in combo) for combo in product(*factor_orders)]
