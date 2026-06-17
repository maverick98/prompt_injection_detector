from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    path = Path("reports/adversarial_history.csv")
    if not path.exists():
        raise SystemExit("Run `pid loop` before plotting adversarial history.")
    frame = pd.read_csv(path)
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(frame["iteration"], frame["attack_success_rate"], marker="o", label="Attack success rate")
    ax1.plot(frame["iteration"], frame["detector_recall"], marker="o", label="Detector recall")
    ax1.plot(frame["iteration"], frame["detector_f1"], marker="o", label="Detector F1")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Score")
    ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.25)
    ax1.legend()
    Path("reports").mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig("reports/adversarial_history.png", dpi=160)
    print("Wrote reports/adversarial_history.png")


if __name__ == "__main__":
    main()

