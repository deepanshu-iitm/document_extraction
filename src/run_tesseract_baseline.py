from pathlib import Path
import pandas as pd
from ocr_utils import extract_text_tesseract
from tqdm import tqdm

from evaluate import compute_cer, compute_wer, compute_accuracy

DATA_ROOT = Path("data/raw")
PREDICTIONS_ROOT = Path("outputs/predictions/baseline")
METRICS_ROOT = Path("outputs/metrics")

GROUPS = [
    "degraded",
    "dense_text",
    "handwritten",
    "printed",
    "receipts",
    "scene_text",
]


def extract_text_from_image(image_path, group):
    return extract_text_tesseract(image_path, group_name=group)


def read_text_file(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read()


def save_prediction(prediction_text, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(prediction_text)


def main():
    PREDICTIONS_ROOT.mkdir(parents=True, exist_ok=True)
    METRICS_ROOT.mkdir(parents=True, exist_ok=True)

    rows = []

    for group in GROUPS:
        group_images_dir = DATA_ROOT / group / "images"
        group_gt_dir = DATA_ROOT / group / "ground_truth"

        image_paths = sorted(group_images_dir.glob("*.png"))

        print(f"\nProcessing group: {group} ({len(image_paths)} images)")

        for image_path in tqdm(image_paths, desc=f"{group}"):
            stem = image_path.stem
            gt_path = group_gt_dir / f"{stem}.txt"

            if not gt_path.exists():
                print(f"WARNING: Missing ground truth for {image_path.name}")
                continue

            prediction = extract_text_from_image(image_path, group)
            reference = read_text_file(gt_path)

            cer_value = compute_cer(prediction, reference)
            wer_value = compute_wer(prediction, reference)
            accuracy_value = compute_accuracy(prediction, reference)

            prediction_output_path = PREDICTIONS_ROOT / group / f"{stem}.txt"
            save_prediction(prediction, prediction_output_path)

            rows.append(
                {
                    "group": group,
                    "image_file": image_path.name,
                    "ground_truth_file": gt_path.name,
                    "prediction_file": str(prediction_output_path),
                    "prediction_text": prediction,
                    "reference_text": reference,
                    "cer": cer_value,
                    "wer": wer_value,
                    "accuracy": accuracy_value,
                }
            )

    results_df = pd.DataFrame(rows)

    if results_df.empty:
        print("No results were generated.")
        return

    detailed_csv_path = METRICS_ROOT / "baseline_detailed_results.csv"
    results_df.to_csv(detailed_csv_path, index=False)

    group_summary_df = (
        results_df.groupby("group")[["cer", "wer", "accuracy"]]
        .mean()
        .reset_index()
        .sort_values("group")
    )

    group_summary_csv_path = METRICS_ROOT / "baseline_group_summary.csv"
    group_summary_df.to_csv(group_summary_csv_path, index=False)

    overall_summary = {
        "overall_cer": results_df["cer"].mean(),
        "overall_wer": results_df["wer"].mean(),
        "overall_accuracy": results_df["accuracy"].mean(),
        "total_images": len(results_df),
    }

    print("\n=== GROUP SUMMARY ===")
    print(group_summary_df.to_string(index=False))

    print("\n=== OVERALL SUMMARY ===")
    for key, value in overall_summary.items():
        print(f"{key}: {value}")

    overall_summary_path = METRICS_ROOT / "baseline_overall_summary.txt"
    with open(overall_summary_path, "w", encoding="utf-8") as f:
        for key, value in overall_summary.items():
            f.write(f"{key}: {value}\n")


if __name__ == "__main__":
    main()