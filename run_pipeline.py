from pathlib import Path
import subprocess
import sys


def run(cmd: list[str]) -> int:
    print(f"\n[RUN] {' '.join(cmd)}")
    return subprocess.call(cmd)


def main():
    project_root = Path(__file__).resolve().parent

    try:
        os.chdir(project_root)
    except Exception:
        pass

    code = run([sys.executable, "src/run_hybrid_baseline.py"])
    if code != 0:
        print(f"\n[ERROR] Hybrid pipeline failed with exit code {code}")
        sys.exit(code)

    metrics_path = Path("outputs/metrics/hybrid_baseline_group_summary.csv")
    if metrics_path.exists():
        import pandas as pd

        df = pd.read_csv(metrics_path)
        print("\n=== FINAL HYBRID GROUP SUMMARY ===")
        print(df.to_string(index=False))
    else:
        print("\n[WARN] Group summary CSV not found; run_hybrid_baseline may not have produced it.")

    overall_path = Path("outputs/metrics/hybrid_baseline_overall_summary.txt")
    if overall_path.exists():
        print("\n=== FINAL HYBRID OVERALL SUMMARY ===")
        print(overall_path.read_text(encoding="utf-8"))
    else:
        print("\n[WARN] Overall summary file not found.")


if __name__ == "__main__":
    import os
    main()