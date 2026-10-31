from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from model import simulate, overlap, energy_and_momentum, exact_collision, contact_info


M1 = 0.17
M2 = 0.17
R1 = 0.0286
R2 = 0.0286
K = 200000
TIME = 0.16
DT = 0.00005

CASES = [
    ("Лобовой удар, равные массы", M1, M2, (0, 0), (0.12, 0), (1, 0), (0, 0)),
    ("Лобовой удар, разные массы", M1, 0.25, (0, 0), (0.12, 0), (1, 0), (0, 0)),
    ("Удар под углом", M1, M2, (0, 0), (0.12, 0.03), (1, 0), (0, 0)),
    ("Оба шара движутся", M1, 0.21, (0, 0), (0.12, -0.018), (0.9, 0.04), (-0.2, -0.03)),
]

PLOTS = Path(__file__).resolve().parent.parent / "docs" / "plots"


def make_case(row, k=K):
    name, m1, m2, p1, p2, v1, v2 = row
    return {"m1": m1, "m2": m2, "r1": R1, "r2": R2, "k": k,
            "p1": p1, "p2": p2, "v1": v1, "v2": v2,
            "time": TIME, "dt": DT}


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    results = []
    for row in CASES:
        p = make_case(row)
        t, state = simulate(p)
        info = contact_info(t, state, p)
        energy, impulse = energy_and_momentum(state, p)
        de = np.max(np.abs(energy - energy[0])) / energy[0]
        dp = np.max(np.linalg.norm(impulse - impulse[0], axis=1))
        if info is None:
            raise RuntimeError("Столкновение не произошло: " + row[0])
        before = state[info["first_index"] - 1]
        theory1, theory2 = exact_collision(p, before)
        final1, final2 = state[-1, 4:6], state[-1, 6:8]
        difference = max(np.linalg.norm(final1 - theory1), np.linalg.norm(final2 - theory2))
        results.append((row[0], p, t, state, info, de, dp, difference,
                        theory1, theory2, final1, final2))
        print("\n" + row[0])
        print("Скорость 1:", np.round(final1, 4), "м/с")
        print("Скорость 2:", np.round(final2, 4), "м/с")
        print(f"Время контакта: {info['duration']:.5f} с")
        print(f"Макс. сжатие: {info['max_overlap'] * 1000:.3f} мм")
        print(f"Ошибка энергии: {de:.2e}")
        print(f"Ошибка импульса: {dp:.2e}")
        print(f"Разница с теорией: {difference:.4f} м/с")

    plt.figure(figsize=(8, 5))
    for name, p, t, s, *rest in results:
        plt.plot(s[:, 0], s[:, 1], label=name + ": 1")
        plt.plot(s[:, 2], s[:, 3], "--", label=name + ": 2")
    plt.xlabel("x, м")
    plt.ylabel("y, м")
    plt.title("Траектории центров шаров")
    plt.axis("equal")
    plt.grid(True)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(PLOTS / "01_paths.png", dpi=170)
    plt.close()

    name, p, t, s, info, *_ = results[0]
    plt.figure(figsize=(8, 4.5))
    plt.plot(t, s[:, 4], label="Скорость первого шара")
    plt.plot(t, s[:, 6], label="Скорость второго шара")
    plt.xlabel("Время, с")
    plt.ylabel("Скорость по x, м/с")
    plt.title("Лобовой удар двух одинаковых шаров")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS / "02_speeds.png", dpi=170)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.plot(t, p["k"] * overlap(s, p))
    plt.xlabel("Время, с")
    plt.ylabel("Сила, Н")
    plt.title("Сила при ударе")
    plt.xlim(info["start"] - .003, info["end"] + .003)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOTS / "03_force.png", dpi=170)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    for name, p, t, s, *_ in results:
        e, _ = energy_and_momentum(s, p)
        plt.plot(t, (e - e[0]) / e[0], label=name)
    plt.xlabel("Время, с")
    plt.ylabel("(E - E0) / E0")
    plt.title("Проверка сохранения энергии")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOTS / "04_energy.png", dpi=170)
    plt.close()

    ks = [100000, 200000, 400000]
    numerical = []
    theoretical = []
    for k in ks:
        p = make_case(CASES[0], k=k)
        t, s = simulate(p)
        duration = contact_info(t, s, p)["duration"]
        reduced_mass = p["m1"] * p["m2"] / (p["m1"] + p["m2"])
        numerical.append(duration * 1000)
        theoretical.append(np.pi * np.sqrt(reduced_mass / k) * 1000)
    plt.figure(figsize=(7, 4.5))
    plt.plot(ks, numerical, "o-", label="Численно")
    plt.plot(ks, theoretical, "s--", label="Теория")
    plt.xlabel("Жёсткость k, Н/м")
    plt.ylabel("Время контакта, мс")
    plt.title("Время удара при разной жёсткости")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS / "05_stiffness.png", dpi=170)
    plt.close()

main()
