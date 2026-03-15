from pathlib import Path
import pandas as pd

RESULTS_CSV = Path("outputs/metrics/hybrid_baseline_detailed_results.csv")
OUTPUT_TXT = Path("outputs/metrics/hybrid_failures_preview.txt")


def main():
    if not RESULTS_CSV.exists():
        print(f"ERROR: File not found -> {RESULTS_CSV}")
        return

    df = pd.read_csv(RESULTS_CSV)

    if df.empty:
        print("ERROR: CSV is empty.")
        return

    worst_df = df.sort_values("accuracy", ascending=True).head(10)

    blocks = []
    blocks.append("=== 10 WORST HYBRID PREDICTIONS ===\n")

    for i, row in enumerate(worst_df.itertuples(index=False), start=1):
        blocks.append(f"#{i}")
        blocks.append(f"Group: {row.group}")
        blocks.append(f"Image: {row.image_file}")
        blocks.append(f"Accuracy: {row.accuracy}")
        blocks.append(f"CER: {row.cer}")
        blocks.append(f"WER: {row.wer}")
        blocks.append("GROUND TRUTH:")
        blocks.append(str(row.reference_text))
        blocks.append("PREDICTION:")
        blocks.append(str(row.prediction_text))
        blocks.append("-" * 80)

    report = "\n".join(blocks)

    print(report)

    OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nSaved preview to: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()