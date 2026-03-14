from pathlib import Path
import pytesseract
from PIL import Image

IMAGE_PATH = Path("data/raw/receipts/images/receipt_000.png")

def main():
    print("Tesseract path:")
    print(pytesseract.pytesseract.tesseract_cmd)

    if not IMAGE_PATH.exists():
        print(f"ERROR: Image not found: {IMAGE_PATH}")
        return

    image = Image.open(IMAGE_PATH)
    text = pytesseract.image_to_string(image, lang="eng")

    print(text)


if __name__ == "__main__":
    main()