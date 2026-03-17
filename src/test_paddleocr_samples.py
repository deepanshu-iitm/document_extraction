from pathlib import Path
from paddleocr import PaddleOCR
from evaluate import compute_accuracy
import cv2

SAMPLES = [
    Path("data/raw/scene_text/images/scene_003.png"),
    Path("data/raw/scene_text/images/scene_004.png"),
    Path("data/raw/scene_text/images/scene_005.png"),
]

ocr = PaddleOCR(
    use_textline_orientation=True,
    lang="en",
)

def extract_text(result):
    lines = []
    for page in result:
        if hasattr(page, "get"):
            texts = page.get("rec_texts", [])
            lines.extend(texts)
    return "\n".join(lines).strip()

def paddle_text_from_path(path: Path) -> str:
    result = ocr.predict(str(path))
    return extract_text(result)

def main():
    for image_path in SAMPLES:
        print(f"\n{'=' * 60}")
        print(f"IMAGE: {image_path}")

        if not image_path.exists():
            print("[NOT FOUND]")
            continue

        gt_path = Path(str(image_path).replace("images", "ground_truth").replace(".png", ".txt"))
        if not gt_path.exists():
            print(f"[GT NOT FOUND] {gt_path}")
            continue
        reference = gt_path.read_text(encoding="utf-8")

        try:
            text_raw = paddle_text_from_path(image_path)
            acc_raw = compute_accuracy(text_raw, reference)

            print("\n--- RAW IMAGE ---")
            print(f"Accuracy: {acc_raw:.2f}")
            print("--- OCR START ---")
            print(text_raw if text_raw else "[EMPTY]")
            print("--- OCR END ---")

            img = cv2.imread(str(image_path))
            if img is None:
                print("\n[WARN] OpenCV could not read image, skipping upscale variant.")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            up2 = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

            tmp_dir = Path("outputs/tmp/scene_text")
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = tmp_dir / (image_path.stem + "_up2.png")
            cv2.imwrite(str(tmp_path), up2)

            text_up2 = paddle_text_from_path(tmp_path)
            acc_up2 = compute_accuracy(text_up2, reference)

            print("\n--- 2x UPSCALED GRAY ---")
            print(f"Accuracy: {acc_up2:.2f}")
            print("--- OCR START ---")
            print(text_up2 if text_up2 else "[EMPTY]")
            print("--- OCR END ---")

        except Exception as e:
            print(f"[ERROR] {e}")

if __name__ == "__main__":
    main()