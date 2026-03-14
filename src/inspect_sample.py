from pathlib import Path
import pandas as pd
import textwrap

RESULTS_CSV = Path("outputs/metrics/baseline_detailed_results.csv")


def pretty_print_block(title, text, width=100):
    print(f"\n=== {title} ===")
    if pd.isna(text):
        print("[EMPTY]")
        return
    wrapped = textwrap.fill(str(text), width=width)
    print(wrapped)


def main():
    if not RESULTS_CSV.exists():
        print(f"ERROR: File not found -> {RESULTS_CSV}")
        return

    df = pd.read_csv(RESULTS_CSV)

    # Change these when you want to inspect another sample
    target_group = "scene_text"
    target_image = "scene_000.png"

    match = df[(df["group"] == target_group) & (df["image_file"] == target_image)]

    if match.empty:
        print(f"Sample not found for group={target_group}, image={target_image}")
        return

    row = match.iloc[0]

    print("=== SAMPLE INFO ===")
    print(f"Group     : {row['group']}")
    print(f"Image     : {row['image_file']}")
    print(f"CER       : {row['cer']}")
    print(f"WER       : {row['wer']}")
    print(f"Accuracy  : {row['accuracy']}")
    print(f"Prediction file: {row['prediction_file']}")

    pretty_print_block("PREDICTION TEXT", row["prediction_text"])
    pretty_print_block("REFERENCE TEXT", row["reference_text"])


if __name__ == "__main__":
    main()