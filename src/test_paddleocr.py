from pathlib import Path
from paddleocr import PaddleOCR

IMAGE_PATH = Path("data/raw/degraded/images/degraded_001.png")

ocr = PaddleOCR(
    use_textline_orientation=True,
    lang="en",
)

def main():
    if not IMAGE_PATH.exists():
        print(f"ERROR: Image not found -> {IMAGE_PATH}")
        return

    result = ocr.predict(str(IMAGE_PATH))

    print("=== PADDLEOCR OUTPUT ===")

    if not result:
        print("[EMPTY]")
        return

    full_text_lines = []

    for page_idx, page in enumerate(result):
        print(f"\n--- PAGE {page_idx} ---")
        print(page)

        texts = []
        scores = []

        if hasattr(page, "get"):
            texts = page.get("rec_texts", [])
            scores = page.get("rec_scores", [])

        if texts:
            for text, score in zip(texts, scores):
                print(f"TEXT: {text} | SCORE: {score:.4f}")
                full_text_lines.append(text)

    print("\n=== COMBINED TEXT ===")
    combined = "\n".join(full_text_lines).strip()
    print(combined if combined else "[EMPTY]")


if __name__ == "__main__":
    main()