import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _ensure_output(path: str = "output") -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_plots(results: dict, output_dir: str = "output"):
    out = _ensure_output(output_dir)

    plt.figure(figsize=(10, 6))
    for name, result in results.items():
        plt.plot(result["time_s"], result["position_error_m"], label=name)
    plt.xlabel("Time since GNSS loss [s]")
    plt.ylabel("3D position error [m]")
    plt.yscale("log")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "position_error.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    for name, result in results.items():
        plt.plot(result["time_s"], np.abs(result["clock_error_ns"]), label=name)
    plt.xlabel("Time since GNSS loss [s]")
    plt.ylabel("Absolute clock bias estimation error [ns]")
    plt.yscale("log")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "clock_error.png", dpi=180)
    plt.close()

    # Visibility is common enough for interpretation; plot the hybrid run.
    hybrid = results["D_hybrid"]
    plt.figure(figsize=(10, 4))
    plt.step(
        hybrid["time_s"],
        hybrid["visible_satellites"],
        where="post",
    )
    plt.xlabel("Time since GNSS loss [s]")
    plt.ylabel("Visible LEO satellites")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "visible_satellites.png", dpi=180)
    plt.close()


def save_summary(results: dict, output_dir: str = "output"):
    out = _ensure_output(output_dir)

    rows = []
    for name, result in results.items():
        pos = result["position_error_m"]
        clk = np.abs(result["clock_error_ns"])
        rows.append(
            {
                "scenario": name,
                "final_position_error_m": float(pos[-1]),
                "position_rmse_m": float(np.sqrt(np.mean(pos**2))),
                "final_abs_clock_error_ns": float(clk[-1]),
                "clock_rmse_ns": float(np.sqrt(np.mean(result["clock_error_ns"] ** 2))),
                "mean_visible_leo": float(np.mean(result["visible_satellites"])),
            }
        )

    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return rows
