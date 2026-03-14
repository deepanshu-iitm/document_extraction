from pathlib import Path
import cv2
import pytesseract

IMAGE_PATH = Path("data/raw/degraded/images/degraded_001.png")
OUTPUT_DIR = Path("outputs/logs/preprocessing_debug")

def ocr_from_image_array(image, config=""):
    return pytesseract.image_to_string(image, lang="eng", config=config)

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
        print("ERROR: OpenCV could not read the image.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    adaptive = cv2.adaptiveThreshold(
        resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )
    median = cv2.medianBlur(resized, 3)

    variants = {
        "gray.png": gray,
        "resized.png": resized,
        "thresh.png": thresh,
        "adaptive.png": adaptive,
        "median.png": median,
    }

    for filename, img in variants.items():
        path = save_image(filename, img)
        print(f"\n=== Variant: {filename} ===")
        print(f"Saved at: {path}")
        text = ocr_from_image_array(img)
        print("--- OCR START ---")
        print(text if text.strip() else "[EMPTY]")
        print("--- OCR END ---")

if __name__ == "__main__":
    main()