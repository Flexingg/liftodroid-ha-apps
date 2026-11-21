from barbell.config import from_dict, with_context
from barbell.inventory import is_loadable, loadings_for_total, side_loadings
from barbell.ocr import TextHit, denominations

CFG = from_dict({"unit": "lb", "bar_weight": 45})


def test_pairs_only_and_totals():
    totals = {l.total(45) for l in side_loadings(CFG.plates)}
    assert 45 in totals and 135 in totals and 225 in totals and 315 in totals
    assert 50 in totals  # 2.5 a side
    assert 47.5 not in totals  # one 2.5 can't go on only one side
    assert all(l.plates == tuple(sorted(l.plates, reverse=True)) for l in side_loadings(CFG.plates))


def test_loadable():
    assert is_loadable(CFG, 185)  # 45+25
    assert not is_loadable(CFG, 187)
    assert loadings_for_total(CFG, 135)[0].plates == (45.0,)


def test_phone_inventory_wins_counts_keeps_sizes():
    cfg = with_context(CFG, "lb", 35, [{"weight": 45, "count": 2}, {"weight": 10, "count": 4}])
    assert cfg.bar_weight == 35
    assert [p.weight for p in cfg.plates] == [45, 10]
    assert cfg.plate_by_weight(45).diameter_mm == 450
    assert not is_loadable(cfg, 35 + 180)  # only one 45 a side now


def hit(t, s=0.95, c=(10.0, 10.0)):
    return TextHit(t, s, c)


def test_denominations_parse():
    ws = [45, 35, 25, 10, 5, 2.5]
    got = denominations([hit("45LB"), hit("REP FITNESS", c=(300, 300)), hit("2.5", c=(100, 100)), hit("25 LBS", c=(200, 200))], ws, "lb", 0.5)
    assert sorted(d.weight for d in got) == [2.5, 25, 45]


def test_denominations_drop_wrong_unit_low_score_and_fix_stencil():
    ws = [45, 35, 25, 20, 10, 5, 2.5]
    assert denominations([hit("20KG")], ws, "lb", 0.5) == []
    assert denominations([hit("45LB", s=0.3)], ws, "lb", 0.5) == []
    assert [d.weight for d in denominations([hit("4S")], ws, "lb", 0.5)] == [45]


def test_same_text_in_two_rotations_counts_once():
    ws = [45]
    got = denominations([hit("45", 0.8, (50, 50)), hit("45LB", 0.9, (52, 51))], ws, "lb", 0.5)
    assert len(got) == 1 and got[0].score == 0.9
