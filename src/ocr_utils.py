from pathlib import Path
import cv2
import pytesseract


def extract_text_tesseract(image_path, group_name=None):
    image = cv2.imread(str(image_path))
    if image is None:
        return ""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Handwritten-specific branch
    if group_name == "handwritten":
        processed = cv2.bilateralFilter(gray, 9, 75, 75)
        config = "--psm 13"
        text = pytesseract.image_to_string(processed, lang="eng", config=config)
        return text

    # Default branch for other groups
    resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    config = "--psm 6"
    text = pytesseract.image_to_string(resized, lang="eng", config=config)
    return text