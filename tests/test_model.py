import numpy as np
import pytest

from model import simulate, validate, overlap, energy_and_momentum, exact_collision, contact_info


def data():
    return {"m1": 0.17, "m2": 0.17, "r1": 0.0286, "r2": 0.0286,
            "k": 200000, "time": 0.16, "dt": 0.00005,
            "p1": (0, 0), "p2": (0.12, 0),
            "v1": (1, 0), "v2": (0, 0)}


def test_equal_masses():
    p = data()
    _, s = simulate(p)
    assert abs(s[-1, 4]) < 1e-5
    assert abs(s[-1, 6] - 1) < 1e-5


def test_different_masses():
    p = data()
    p["m2"] = 0.25
    _, s = simulate(p)
    expected1 = (p["m1"] - p["m2"]) / (p["m1"] + p["m2"])
    expected2 = 2 * p["m1"] / (p["m1"] + p["m2"])
    assert abs(s[-1, 4] - expected1) < 2e-5
    assert abs(s[-1, 6] - expected2) < 2e-5


@pytest.mark.parametrize("side", [0, 0.02, -0.03])
def test_conservation(side):
    p = data()
    p["p2"] = (0.12, side)
    _, s = simulate(p)
    e, mom = energy_and_momentum(s, p)
    assert np.max(np.abs(e - e[0])) / e[0] < 1e-6
    assert np.max(np.linalg.norm(mom - mom[0], axis=1)) < 1e-9


def test_both_moving():
    p = data()
    p["m2"] = 0.21
    p["p2"] = (0.12, -0.018)
    p["v1"] = (0.9, 0.04)
    p["v2"] = (-0.2, -0.03)
    t, s = simulate(p)
    e, mom = energy_and_momentum(s, p)
    assert contact_info(t, s, p) is not None
    assert np.max(np.abs(e - e[0])) / e[0] < 1e-6
    assert np.max(np.linalg.norm(mom - mom[0], axis=1)) < 1e-9


def test_contact_duration():
    p = data()
    t, s = simulate(p)
    actual = contact_info(t, s, p)["duration"]
    mu = p["m1"] * p["m2"] / (p["m1"] + p["m2"])
    expected = np.pi * np.sqrt(mu / p["k"])
    assert abs(actual - expected) / expected < 0.06


def test_max_compression():
    p = data()
    t, s = simulate(p)
    actual = contact_info(t, s, p)["max_overlap"]
    mu = p["m1"] * p["m2"] / (p["m1"] + p["m2"])
    expected = np.sqrt(mu / p["k"])
    assert abs(actual - expected) / expected < 0.01


def test_exact_vector_velocity():
    p = data()
    t, s = simulate(p)
    i = contact_info(t, s, p)["first_index"]
    v1, v2 = exact_collision(p, s[i - 1])
    assert np.linalg.norm(s[-1, 4:6] - v1) < 0.001
    assert np.linalg.norm(s[-1, 6:8] - v2) < 0.001


def test_increasing_stiffness_shortens_contact():
    p = data()
    p["k"] = 100000
    t1, s1 = simulate(p)
    a = contact_info(t1, s1, p)["duration"]
    p["k"] = 400000
    t2, s2 = simulate(p)
    b = contact_info(t2, s2, p)["duration"]
    assert a > b


def test_no_contact():
    p = data()
    p["p2"] = (0.12, 0.09)
    t, s = simulate(p)
    assert contact_info(t, s, p) is None
    assert np.all(overlap(s, p) == 0)


@pytest.mark.parametrize("field,value", [
    ("m1", -0.17), ("m2", 0), ("r1", 0.01),
    ("r2", -0.02), ("k", -100000), ("k", 100),
    ("time", 0), ("time", 10), ("dt", 0),
    ("dt", 0.001), ("m1", float("nan")),
    ("k", float("inf")), ("m1", "heavy"),
])
def test_invalid_numbers(field, value):
    p = data()
    p[field] = value
    with pytest.raises(ValueError):
        validate(p)


def test_initial_overlap():
    p = data()
    p["p2"] = (0.01, 0)
    with pytest.raises(ValueError):
        validate(p)


def test_bad_density():
    p = data()
    p["m1"] = 0.08
    p["r1"] = 0.036
    with pytest.raises(ValueError):
        validate(p)


def test_bad_velocity():
    p = data()
    p["v1"] = (4, 0)
    with pytest.raises(ValueError):
        validate(p)


def test_bad_position():
    p = data()
    p["p1"] = (2, 0)
    with pytest.raises(ValueError):
        validate(p)


def test_bad_vector():
    p = data()
    p["v2"] = (0, 0, 0)
    with pytest.raises(ValueError):
        validate(p)


def test_large_deformation():
    p = data()
    p["k"] = 50000
    p["v1"] = (2.5, 0)
    with pytest.raises(ValueError):
        validate(p)


def test_theory_for_off_axis_impact():
    p = data()
    p["p2"] = (0.12, 0.03)
    t, s = simulate(p)
    i = contact_info(t, s, p)["first_index"]
    v1, v2 = exact_collision(p, s[i - 1])
    assert np.linalg.norm(s[-1, 4:6] - v1) < 0.03
    assert np.linalg.norm(s[-1, 6:8] - v2) < 0.03
