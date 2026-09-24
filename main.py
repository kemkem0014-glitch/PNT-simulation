from config import CFG
from plots import save_plots, save_summary
from simulation import run_all


def main():
    print("Hybrid LEO-PNT simulation")
    print("=" * 32)
    print(f"Duration: {CFG.duration_s:.0f} s")
    print(f"LEO satellites: {CFG.leo_satellites}")
    print(f"Iridium timing sigma: {CFG.iridium_time_sigma_ns:.1f} ns")
    print(f"Doppler sigma: {CFG.doppler_sigma_hz:.1f} Hz")
    print()

    results = run_all(CFG)
    save_plots(results)
    rows = save_summary(results)

    print("Scenario summary")
    print("-" * 80)
    for row in rows:
        print(
            f'{row["scenario"]:18s} '
            f'pos_RMSE={row["position_rmse_m"]:10.3f} m  '
            f'final_pos={row["final_position_error_m"]:10.3f} m  '
            f'clock_RMSE={row["clock_rmse_ns"]:10.3f} ns'
        )

    print()
    print("Saved results to ./output/")


if __name__ == "__main__":
    main()
