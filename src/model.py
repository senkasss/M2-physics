import numpy as np
from scipy.integrate import solve_ivp


def validate(p):
    for key, low, high in [
        ("m1", 0.08, 0.40), ("m2", 0.08, 0.40),
        ("r1", 0.024, 0.036), ("r2", 0.024, 0.036),
        ("k", 50000, 1000000), ("time", 0.08, 1.0),
        ("dt", 0.000002, 0.0002),
    ]:
        value = p[key]
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)) or not np.isfinite(value):
            raise ValueError(f"Неверное значение: {key}")
        if not low <= value <= high:
            raise ValueError(f"{key} должно быть от {low} до {high}")

    for key in ("p1", "p2", "v1", "v2"):
        try:
            a = np.asarray(p[key], dtype=float)
        except (TypeError, ValueError):
            raise ValueError(f"Неверное значение: {key}") from None
        if a.shape != (2,) or not np.all(np.isfinite(a)):
            raise ValueError(f"Неверное значение: {key}")
        if key[0] == "p" and (abs(a[0]) > 1.2 or abs(a[1]) > 0.6):
            raise ValueError("Начальная точка вне рабочей области")
        if key[0] == "v" and np.linalg.norm(a) > 3:
            raise ValueError("Скорость не должна превышать 3 м/с")

    r1, r2 = p["r1"], p["r2"]
    m1, m2 = p["m1"], p["m2"]
    for m, r in ((m1, r1), (m2, r2)):
        density = m / (4 * np.pi * r**3 / 3)
        if not 600 <= density <= 9000:
            raise ValueError("Слишком странная плотность для бильярдного шара)")

    if np.linalg.norm(np.array(p["p2"]) - np.array(p["p1"])) <= r1 + r2:
        raise ValueError("Шары не должны пересекаться в начальный момент")

    reduced_mass = m1 * m2 / (m1 + m2)
    relative_speed = np.linalg.norm(np.array(p["v1"]) - np.array(p["v2"]))
    estimated_overlap = relative_speed * np.sqrt(reduced_mass / p["k"])
    if estimated_overlap > 0.08 * min(r1, r2):
        raise ValueError("Слишком большая возможная деформация шара: увеличьте k или уменьшите скорость")
    contact_time = np.pi * np.sqrt(reduced_mass / p["k"])
    if p["dt"] > contact_time / 20:
        raise ValueError("Слишком большой шаг dt для выбранной жёсткости")


def overlap(state, p):
    state = np.asarray(state)
    dist = np.hypot(state[..., 2] - state[..., 0], state[..., 3] - state[..., 1])
    return np.maximum(p["r1"] + p["r2"] - dist, 0)


def equations(t, y, p):
    x1, y1, x2, y2, vx1, vy1, vx2, vy2 = y
    dx, dy = x2 - x1, y2 - y1
    dist = np.hypot(dx, dy)
    compression = max(p["r1"] + p["r2"] - dist, 0)
    if dist > 0:
        fx = p["k"] * compression * dx / dist
        fy = p["k"] * compression * dy / dist
    else:
        fx, fy = 0, 0
    return [vx1, vy1, vx2, vy2,
            -fx / p["m1"], -fy / p["m1"], fx / p["m2"], fy / p["m2"]]


def simulate(p):
    validate(p)
    y0 = [*p["p1"], *p["p2"], *p["v1"], *p["v2"]]
    n = int(np.ceil(p["time"] / p["dt"]))
    t = np.linspace(0, p["time"], n + 1)
    sol = solve_ivp(equations, (0, p["time"]), y0, args=(p,),
                    t_eval=t, method="RK45", max_step=p["dt"],
                    rtol=1e-9, atol=1e-11)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.t, sol.y.T


def energy_and_momentum(state, p):
    a = np.asarray(state)
    v1 = a[..., 4:6]
    v2 = a[..., 6:8]
    kinetic = p["m1"] * np.sum(v1**2, axis=-1) / 2
    kinetic += p["m2"] * np.sum(v2**2, axis=-1) / 2
    elastic = p["k"] * overlap(a, p)**2 / 2
    impulse = p["m1"] * v1 + p["m2"] * v2
    return kinetic + elastic, impulse


def exact_collision(p, state):
    a = np.asarray(state)
    n = a[2:4] - a[0:2]
    n = n / np.linalg.norm(n)
    v1 = a[4:6]
    v2 = a[6:8]
    speed = np.dot(v1 - v2, n)
    m1, m2 = p["m1"], p["m2"]
    v1_after = v1 - 2 * m2 / (m1 + m2) * speed * n
    v2_after = v2 + 2 * m1 / (m1 + m2) * speed * n
    return v1_after, v2_after


def contact_info(t, state, p):
    a = np.asarray(state)
    dist = np.hypot(a[:, 2] - a[:, 0], a[:, 3] - a[:, 1])
    gap = p["r1"] + p["r2"] - dist
    inside = np.where(gap > 1e-10)[0]
    if not len(inside):
        return None
    first = inside[0]
    last = inside[-1]
    if last == len(t) - 1:
        raise ValueError("Шары ещё соприкасаются: увеличьте время моделирования, чтобы увидеть более внятный результат")
    start = t[first - 1] - gap[first - 1] * (t[first] - t[first - 1]) / (gap[first] - gap[first - 1])
    end = t[last] - gap[last] * (t[last + 1] - t[last]) / (gap[last + 1] - gap[last])
    max_overlap = np.max(gap)
    return {"start": start, "end": end, "duration": end - start,
            "max_overlap": max_overlap, "max_force": p["k"] * max_overlap,
            "first_index": first, "last_index": last}
