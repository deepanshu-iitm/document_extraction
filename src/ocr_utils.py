from pathlib import Path
import cv2
import pytesseract
from paddleocr import PaddleOCR

_PADDLE_OCR = None


def get_paddle_ocr():
    global _PADDLE_OCR

    if _PADDLE_OCR is None:
        _PADDLE_OCR = PaddleOCR(
            use_doc_orientation_classify=True,
            use_doc_unwarping=True,
            use_textline_orientation=True,
        )

    return _PADDLE_OCR


def extract_text_paddle(image_path):
    ocr = get_paddle_ocr()
    results = ocr.predict(str(image_path))

    texts = []

    for page in results:
        rec_texts = page.get("rec_texts", [])
        for text in rec_texts:
            if text and str(text).strip():
                texts.append(str(text).strip())

    return "\n".join(texts)


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