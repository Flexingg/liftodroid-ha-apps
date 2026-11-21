"""Turns every cue into one ranked answer.

Each loading the inventory allows is scored by how well it explains what the camera saw:
  * structure: its predicted height profile (plate diameters along the sleeve) against the observed one, and its
    stack thickness against the observed stack length;
  * lettering: denominations read on the plates must be on the bar (the outermost plate, which faces the camera
    most, counts most);
  * expectation: a small nudge towards the weight the program prescribed, never enough to override the camera.
Scores become probabilities over loadings, then over totals (35+10 and 45 make the same total).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from .config import Config
from .inventory import Loading, side_loadings, stack_thickness_mm
from .ocr import Denomination
from .slabs import Profile


@dataclass(frozen=True)
class SideEvidence:
    profile: Profile | None
    # Denominations read on this sleeve, each with its position along the sleeve (mm from the collar) if known.
    denominations: tuple[tuple[Denomination, float | None], ...] = ()


@dataclass
class Candidate:
    loading: Loading
    log_score: float
    probability: float = 0.0


@dataclass
class SideResult:
    loading: Loading
    confidence: float
    alternatives: list[tuple[tuple[float, ...], float]]
    structural: bool
    lettering: bool


@dataclass
class Reading:
    detected_weight: float | None
    unit: str
    confidence: float
    per_side_breakdown: list[float]
    per_side_b: list[float] | None
    symmetric: bool | None
    raw_ocr_matches: list[dict]
    needs_confirmation: bool
    status: str
    reasons: list[str] = field(default_factory=list)
    alternatives: list[dict] = field(default_factory=list)
    frame_timestamp: float | None = None
    last_motion_ts: float | None = None
    context: dict | None = None

    def to_payload(self) -> dict:
        return asdict(self)


def _predicted_heights(cfg: Config, loading: Loading, x_mm: np.ndarray) -> np.ndarray | None:
    out = np.zeros_like(x_mm, dtype=np.float32)
    pos = 0.0
    for w in loading.plates:
        p = cfg.plate_by_weight(w)
        if p is None or p.thickness_mm <= 0 or p.diameter_mm <= 0:
            return None  # sizes unknown: no structural prediction
        out[(x_mm >= pos) & (x_mm < pos + p.thickness_mm)] = p.diameter_mm
        pos += p.thickness_mm
    return out


def structural_log_likelihood(cfg: Config, loading: Loading, prof: Profile) -> float | None:
    if prof is None:
        return None
    pred_len = stack_thickness_mm(cfg, loading)
    span = max(prof.stack_length_mm, pred_len) + 20.0
    x = np.arange(0.0, span, 2.0)
    pred = _predicted_heights(cfg, loading, x)
    if pred is None:
        return None
    obs = prof.height_at_mm(x)
    ok = prof.valid_at_mm(x)
    if ok.sum() < max(3, len(x) // 4):
        return None  # mostly glare or hidden: the profile says nothing
    # Mismatch AREA (mm x mm over readable columns), in units of a 10 mm sliver of the largest plate. An area, not
    # an average, so one wrong plate costs the same whether the stack is one plate or five.
    scale = max((p.diameter_mm for p in cfg.plates), default=450.0)
    err = float(np.abs(pred[ok] - obs[ok]).sum() * 2.0 / (scale * 10.0))
    # Stack length error: tolerant to a few mm of slop at the outer edge plus a little per plate.
    th = cfg.thresholds
    sigma = th.length_tolerance_mm + 0.05 * max(pred_len, 1.0)
    len_err = (prof.stack_length_mm - pred_len) / sigma
    return -th.shape_weight * err - 0.5 * len_err * len_err


def lettering_log_likelihood(loading: Loading, side: SideEvidence) -> float:
    ll = 0.0
    for d, pos_mm in side.denominations:
        present = loading.count(d.weight) > 0
        ll += d.score * (math.log(0.9) if present else math.log(0.08))
        if present and loading.outermost is not None and abs(loading.outermost - d.weight) < 1e-6 and pos_mm is not None:
            # Read near the outer end of the stack: that's the plate facing out.
            ll += 0.4 * d.score
    return ll


def score_side(cfg: Config, side: SideEvidence, expected_side: float | None, candidates: list[Loading] | None = None) -> SideResult | None:
    loadings = candidates if candidates is not None else side_loadings(cfg.plates)
    has_struct = side.profile is not None and side.profile.valid.any()
    has_text = bool(side.denominations)
    if not loadings:
        return None
    scored: list[Candidate] = []
    any_struct = False
    for l in loadings:
        s = structural_log_likelihood(cfg, l, side.profile) if has_struct else None
        any_struct |= s is not None
        ll = (s or 0.0) + lettering_log_likelihood(l, side)
        if expected_side is not None and abs(l.side_weight - expected_side) < 1e-6:
            ll += math.log(2.0)
        ll -= 0.03 * len(l.plates)  # among equals, the fewer-plates loading is how people load
        scored.append(Candidate(l, ll))
    if not any_struct and not has_text:
        return None
    m = max(c.log_score for c in scored)
    z = sum(math.exp(c.log_score - m) for c in scored)
    for c in scored:
        c.probability = math.exp(c.log_score - m) / z
    by_side: dict[float, float] = {}
    for c in scored:
        by_side[round(c.loading.side_weight, 3)] = by_side.get(round(c.loading.side_weight, 3), 0.0) + c.probability
    best_side = max(by_side, key=by_side.get)
    best = max((c for c in scored if round(c.loading.side_weight, 3) == best_side), key=lambda c: c.probability)
    alts = sorted(((round(c.loading.side_weight, 3), p) for c in scored for p in [c.probability]), key=lambda t: -t[1])
    seen = set()
    alternatives = []
    for sw, _ in alts:
        if sw in seen:
            continue
        seen.add(sw)
        alternatives.append(((sw,), by_side[sw]))
        if len(alternatives) == 3:
            break
    return SideResult(best.loading, by_side[best_side], alternatives, any_struct, has_text)


def fuse(cfg: Config, side_a: SideEvidence, side_b: SideEvidence | None = None, expected_total: float | None = None) -> Reading:
    th = cfg.thresholds
    expected_side = (expected_total - cfg.bar_weight) / 2 if expected_total is not None else None
    a = score_side(cfg, side_a, expected_side)
    raw = [
        {"weight": d.weight, "score": round(d.score, 3), "text": d.text, "sleeve": "a", "position_mm": None if p is None else round(p, 1)}
        for d, p in side_a.denominations
    ]
    if side_b is not None:
        raw += [
            {"weight": d.weight, "score": round(d.score, 3), "text": d.text, "sleeve": "b", "position_mm": None if p is None else round(p, 1)}
            for d, p in side_b.denominations
        ]
    if a is None:
        return Reading(None, cfg.unit, 0.0, [], None, None, raw, True, "no_evidence", ["The sleeve couldn't be read: nothing to go on."])
    reasons: list[str] = []
    b = score_side(cfg, side_b, expected_side) if side_b is not None else None
    side_b_weight = b.loading.side_weight if b is not None else a.loading.side_weight
    total = cfg.bar_weight + a.loading.side_weight + side_b_weight
    symmetric = None if b is None else abs(a.loading.side_weight - b.loading.side_weight) < 1e-6
    confidence = a.confidence if b is None else a.confidence * b.confidence
    if symmetric is False:
        reasons.append(f"Sides don't match: {a.loading.side_weight:g} vs {b.loading.side_weight:g} {cfg.unit} per side.")
    if not a.structural:
        reasons.append("Read from lettering only: plate sizes weren't measurable.")
    if not a.lettering:
        reasons.append("No lettering read: from plate sizes only.")
    if side_a.profile is not None and side_a.profile.glare_fraction > 0.05:
        reasons.append("Glare on the plates.")
    if expected_total is not None and abs(total - expected_total) > 1e-6:
        reasons.append(f"Differs from the planned {expected_total:g} {cfg.unit}.")
    needs = confidence < th.auto_apply_confidence or symmetric is False or not (a.structural or a.lettering)
    status = "asymmetric" if symmetric is False else ("ok" if not needs else "low_confidence")
    alternatives = [
        {"total": round(cfg.bar_weight + 2 * sw[0], 3), "per_side": sw[0], "probability": round(p, 3)} for sw, p in a.alternatives
    ]
    return Reading(
        detected_weight=round(total, 3),
        unit=cfg.unit,
        confidence=round(confidence, 3),
        per_side_breakdown=list(a.loading.plates),
        per_side_b=list(b.loading.plates) if b is not None else None,
        symmetric=symmetric,
        raw_ocr_matches=raw,
        needs_confirmation=needs,
        status=status,
        reasons=reasons,
        alternatives=alternatives,
    )
