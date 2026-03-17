from pathlib import Path

import cv2
from ocr_utils import extract_text_paddle
from evaluate import compute_cer, compute_wer, compute_accuracy

DATA_ROOT = Path("data/raw/printed")
IMAGES_DIR = DATA_ROOT / "images"
GT_DIR = DATA_ROOT / "ground_truth"


def build_variants(img_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_2x = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    bilateral_2x = cv2.bilateralFilter(gray_2x, 9, 75, 75)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_2x = clahe.apply(gray_2x)

    return {
        "raw": img_path,         
        "gray": gray,
        "gray_2x": gray_2x,
        "gray_2x_bilateral": bilateral_2x,
        "gray_2x_clahe": clahe_2x,
    }


def run_paddle_on_array(img_array, stem: str, variant_name: str) -> str:
    tmp_dir = Path("outputs/tmp/printed_paddle")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_path = tmp_dir / f"{stem}_{variant_name}.png"
    cv2.imwrite(str(out_path), img_array)
    return extract_text_paddle(out_path)


def main():
    if not IMAGES_DIR.exists() or not GT_DIR.exists():
        print("ERROR: printed images or ground_truth folder missing")
        return

    image_paths = sorted(IMAGES_DIR.glob("*.png"))
    if not image_paths:
        print("No printed images found.")
        return

    print(f"Found {len(image_paths)} printed images")

    stats = {
        "raw": [],
        "gray": [],
        "gray_2x": [],
        "gray_2x_bilateral": [],
        "gray_2x_clahe": [],
    }

    for img_path in image_paths:
        stem = img_path.stem
        gt_path = GT_DIR / f"{stem}.txt"
        if not gt_path.exists():
            print(f"[SKIP] GT missing for {img_path.name}")
            continue

        reference = gt_path.read_text(encoding="utf-8")

        variants = build_variants(img_path)

        pred_raw = extract_text_paddle(img_path)
        acc_raw = compute_accuracy(pred_raw, reference)
        stats["raw"].append(acc_raw)

        for vname in ["gray", "gray_2x", "gray_2x_bilateral", "gray_2x_clahe"]:
            pred = run_paddle_on_array(variants[vname], stem, vname)
            acc = compute_accuracy(pred, reference)
            stats[vname].append(acc)

        print(
            f"{img_path.name}: "
            f"raw={acc_raw:.2f}, "
            f"gray={stats['gray'][-1]:.2f}, "
            f"gray_2x={stats['gray_2x'][-1]:.2f}"
            f"gray_2x_bilateral={stats['gray_2x_bilateral'][-1]:.2f}, "
            f"gray_2x_clahe={stats['gray_2x_clahe'][-1]:.2f}"
        )

    print("\n=== PRINTED PADDLE PREPROCESSING SUMMARY ===")
    for vname, values in stats.items():
        if not values:
            continue
        mean_acc = sum(values) / len(values)
        print(f"{vname}: mean_accuracy={mean_acc:.2f} over {len(values)} images")


if __name__ == "__main__":
    main()