from pathlib import Path
import cv2

from ocr_utils import extract_text_paddle

IMAGE_NAMES = [
    "degraded_001.png",
    "degraded_002.png",
    "degraded_003.png",
    "degraded_004.png",
    "degraded_005.png",
]

IMAGE_DIR = Path("data/raw/degraded/images")
OUTPUT_DIR = Path("outputs/preprocessed/degraded_paddle")


def build_variants(image_path: Path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    upscale_2x = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    bilateral_2x = cv2.bilateralFilter(upscale_2x, 9, 75, 75)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_2x = clahe.apply(upscale_2x)

    return {
        "gray": gray,
        "upscale_2x": upscale_2x,
        "bilateral_2x": bilateral_2x,
        "clahe_2x": clahe_2x,
    }


def main():
    if not IMAGE_DIR.exists():
        print(f"ERROR: Image directory not found -> {IMAGE_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for image_name in IMAGE_NAMES:
        image_path = IMAGE_DIR / image_name

        if not image_path.exists():
            print(f"SKIPPING: {image_path} not found")
            continue

        print("\n" + "=" * 100)
        print(f"IMAGE: {image_name}")

        variants = build_variants(image_path)

        image_output_dir = OUTPUT_DIR / image_path.stem
        image_output_dir.mkdir(parents=True, exist_ok=True)

        for variant_name, variant_img in variants.items():
            output_path = image_output_dir / f"{variant_name}.png"
            cv2.imwrite(str(output_path), variant_img)

            text = extract_text_paddle(output_path)

            print("\n" + "-" * 80)
            print(f"VARIANT: {variant_name}")
            print(f"FILE: {output_path}")
            print("--- OCR OUTPUT START ---")
            print(text if text.strip() else "[EMPTY]")
            print("--- OCR OUTPUT END ---")


if __name__ == "__main__":
    main()