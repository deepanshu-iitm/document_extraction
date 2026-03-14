from pathlib import Path
import cv2
import pytesseract

IMAGE_PATH = Path("data/raw/degraded/images/degraded_001.png")
OUTPUT_DIR = Path("outputs/logs/degraded_debug")

PSM_MODES = [6, 11, 13]


def ocr(img, config=""):
    return pytesseract.image_to_string(img, lang="eng", config=config)


def save_image(name, image):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / name
    cv2.imwrite(str(out_path), image)
    return out_path


def main():
    if not IMAGE_PATH.exists():
        print(f"ERROR: Image not found -> {IMAGE_PATH}")
        return

    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        print("ERROR: Could not read image")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_2x = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    resized_3x = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    median = cv2.medianBlur(resized_2x, 3)
    bilateral = cv2.bilateralFilter(resized_2x, 9, 75, 75)
    thresh = cv2.threshold(resized_2x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    adaptive = cv2.adaptiveThreshold(
        resized_2x, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )

    variants = {
        "gray.png": gray,
        "resized_2x.png": resized_2x,
        "resized_3x.png": resized_3x,
        "median.png": median,
        "bilateral.png": bilateral,
        "thresh.png": thresh,
        "adaptive.png": adaptive,
    }

    for filename, img in variants.items():
        save_image(filename, img)

        for psm in PSM_MODES:
            config = f"--psm {psm}"
            text = ocr(img, config=config)

            print(f"\n=== Variant: {filename} | PSM: {psm} ===")
            print("--- OCR START ---")
            print(text if text.strip() else "[EMPTY]")
            print("--- OCR END ---")


if __name__ == "__main__":
    main()