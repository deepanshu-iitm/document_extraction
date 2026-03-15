from pathlib import Path
from paddleocr import PaddleOCR

SAMPLES = [
    Path("data/raw/printed/images/printed_000.png"),
    Path("data/raw/receipts/images/receipt_000.png"),
    Path("data/raw/scene_text/images/scene_000.png"),
    Path("data/raw/handwritten/images/handwritten_000.png"),
    Path("data/raw/degraded/images/degraded_001.png"),
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

def main():
    for image_path in SAMPLES:
        print(f"\n{'=' * 60}")
        print(f"IMAGE: {image_path}")

        if not image_path.exists():
            print("[NOT FOUND]")
            continue

        try:
            result = ocr.predict(str(image_path))
            text = extract_text(result)

            print("--- OCR START ---")
            print(text if text else "[EMPTY]")
            print("--- OCR END ---")

        except Exception as e:
            print(f"[ERROR] {e}")

if __name__ == "__main__":
    main()