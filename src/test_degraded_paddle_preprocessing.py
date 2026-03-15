from pathlib import Path
import cv2

from ocr_utils import extract_text_paddle

IMAGE_DIR = Path("data/raw/degraded/images")
OUTPUT_DIR = Path("outputs/preprocessed/degraded_paddle")


def preprocess_for_degraded(image_path: Path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    upscaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    denoised = cv2.GaussianBlur(upscaled, (3, 3), 0)

    processed = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15,
    )

    return processed


def main():
    if not IMAGE_DIR.exists():
        print(f"ERROR: Image directory not found -> {IMAGE_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(IMAGE_DIR.glob("*.png"))

    if not image_paths:
        print(f"ERROR: No PNG files found in {IMAGE_DIR}")
        return

    for image_path in image_paths:
        processed = preprocess_for_degraded(image_path)

        output_path = OUTPUT_DIR / image_path.name
        cv2.imwrite(str(output_path), processed)

        text = extract_text_paddle(output_path)

        print("\n" + "=" * 80)
        print(f"IMAGE: {image_path.name}")
        print(f"PREPROCESSED FILE: {output_path}")
        print("--- OCR OUTPUT START ---")
        print(text if text.strip() else "[EMPTY]")
        print("--- OCR OUTPUT END ---")


if __name__ == "__main__":
    main()