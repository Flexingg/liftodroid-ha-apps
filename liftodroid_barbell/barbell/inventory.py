"""What the plates you own can put on one side of the bar."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from .config import Config, Plate


@dataclass(frozen=True)
class Loading:
    """Plates on ONE side, inner (heaviest, against the collar) to outer."""

    plates: tuple[float, ...]

    @property
    def side_weight(self) -> float:
        return sum(self.plates)

    def total(self, bar: float) -> float:
        return bar + 2 * self.side_weight

    @property
    def outermost(self) -> float | None:
        return self.plates[-1] if self.plates else None

    def count(self, w: float) -> int:
        return sum(1 for p in self.plates if abs(p - w) < 1e-6)


def side_loadings(plates: tuple[Plate, ...], max_plates_per_side: int = 10) -> list[Loading]:
    """Every per-side loading the inventory allows (pairs only: one plate per side needs two owned).

    Plates go heaviest-first, the way a bar is loaded. Ordered by side weight, then fewest plates.
    """
    kinds = sorted((p for p in plates if p.count >= 2), key=lambda p: -p.weight)
    ranges = [range(p.count // 2 + 1) for p in kinds]
    out: list[Loading] = []
    for counts in itertools.product(*ranges):
        n = sum(counts)
        if n > max_plates_per_side:
            continue
        side: list[float] = []
        for p, c in zip(kinds, counts):
            side.extend([p.weight] * c)
        out.append(Loading(tuple(side)))
    out.sort(key=lambda l: (l.side_weight, len(l.plates)))
    return out


def loadings_for_total(cfg: Config, total: float) -> list[Loading]:
    return [l for l in side_loadings(cfg.plates) if abs(l.total(cfg.bar_weight) - total) < 1e-6]


def is_loadable(cfg: Config, total: float) -> bool:
    return bool(loadings_for_total(cfg, total))


def stack_thickness_mm(cfg: Config, loading: Loading) -> float:
    return sum((cfg.plate_by_weight(w).thickness_mm if cfg.plate_by_weight(w) else 0.0) for w in loading.plates)
