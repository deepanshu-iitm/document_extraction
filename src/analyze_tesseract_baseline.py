from pathlib import Path
import pandas as pd

RESULTS_CSV = Path("outputs/metrics/baseline_detailed_results.csv")

def main():
    if not RESULTS_CSV.exists():
        print(f"ERROR: File not found -> {RESULTS_CSV}")
        return

    df = pd.read_csv(RESULTS_CSV)

    print("\n=== OVERALL DATASET INFO ===")
    print(f"Total rows: {len(df)}")
    print(f"Groups: {sorted(df['group'].unique().tolist())}")

    print("\n=== GROUP-WISE SUMMARY ===")
    group_summary = (
        df.groupby("group")[["cer", "wer", "accuracy"]]
        .mean()
        .sort_values("accuracy", ascending=False)
    )
    print(group_summary.to_string())

    print("\n=== TOP 5 BEST SAMPLES ===")
    best_samples = df.sort_values("accuracy", ascending=False).head(5)
    print(best_samples[["group", "image_file", "accuracy", "cer", "wer"]].to_string(index=False))

    print("\n=== TOP 5 WORST SAMPLES ===")
    worst_samples = df.sort_values("accuracy", ascending=True).head(5)
    print(worst_samples[["group", "image_file", "accuracy", "cer", "wer"]].to_string(index=False))

    analysis_output_path = Path("outputs/metrics/baseline_error_analysis.txt")
    with open(analysis_output_path, "w", encoding="utf-8") as f:
        f.write("=== GROUP-WISE SUMMARY ===\n")
        f.write(group_summary.to_string())
        f.write("\n\n=== TOP 5 BEST SAMPLES ===\n")
        f.write(best_samples[["group", "image_file", "accuracy", "cer", "wer"]].to_string(index=False))
        f.write("\n\n=== TOP 5 WORST SAMPLES ===\n")
        f.write(worst_samples[["group", "image_file", "accuracy", "cer", "wer"]].to_string(index=False))

    print(f"\nSaved analysis to: {analysis_output_path}")


if __name__ == "__main__":
    main()