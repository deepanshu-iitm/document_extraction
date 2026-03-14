from pathlib import Path
import cv2
import pytesseract

IMAGE_PATH = Path("data/raw/scene_text/images/scene_000.png")

PSM_MODES = [3, 4, 6, 7, 11, 12, 13]


def main():
    if not IMAGE_PATH.exists():
        print(f"ERROR: Image not found -> {IMAGE_PATH}")
        return

    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        print("ERROR: OpenCV could not read the image.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    for psm in PSM_MODES:
        config = f"--psm {psm}"
        text = pytesseract.image_to_string(resized, lang="eng", config=config)

        print(f"\n=== PSM {psm} ===")
        print("--- OCR START ---")
        print(text if text.strip() else "[EMPTY]")
        print("--- OCR END ---")


if __name__ == "__main__":
    main()