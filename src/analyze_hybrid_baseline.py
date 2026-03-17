from pathlib import Path
import pandas as pd

RESULTS_CSV = Path("outputs/metrics/hybrid_baseline_detailed_results.csv")
OUTPUT_TXT = Path("outputs/metrics/hybrid_error_analysis.txt")


def main():
    if not RESULTS_CSV.exists():
        print(f"ERROR: Results file not found -> {RESULTS_CSV}")
        return

    df = pd.read_csv(RESULTS_CSV)

    if df.empty:
        print("ERROR: Results CSV is empty.")
        return

    lines = []

    lines.append("=== OVERALL DATASET INFO ===")
    lines.append(f"Total rows: {len(df)}")
    lines.append(f"Groups: {sorted(df['group'].unique().tolist())}")
    lines.append("")

    lines.append("=== GROUP-WISE SUMMARY ===")
    group_summary = (
        df.groupby("group")[["cer", "wer", "accuracy"]]
        .mean()
        .sort_values("accuracy", ascending=False)
    )
    lines.append(group_summary.to_string())
    lines.append("")

    lines.append("=== TOP 5 BEST SAMPLES ===")
    best_samples = df.sort_values("accuracy", ascending=False).head(5)[
        ["group", "image_file", "accuracy", "cer", "wer"]
    ]
    lines.append(best_samples.to_string(index=False))
    lines.append("")

    lines.append("=== TOP 5 WORST SAMPLES ===")
    worst_samples = df.sort_values("accuracy", ascending=True).head(5)[
        ["group", "image_file", "accuracy", "cer", "wer"]
    ]
    lines.append(worst_samples.to_string(index=False))
    lines.append("")

    report_text = "\n".join(lines)

    print(report_text)

    OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\nSaved analysis to: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()