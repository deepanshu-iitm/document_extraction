import cv2
import pytesseract
from paddleocr import PaddleOCR


_PADDLE_OCR = None


def get_paddle_ocr():
    global _PADDLE_OCR

    if _PADDLE_OCR is None:
        _PADDLE_OCR = PaddleOCR(
            use_textline_orientation=True,
            lang="en",
        )

    return _PADDLE_OCR


def extract_text_paddle(image_path):
    try:
        ocr = get_paddle_ocr()
        result = ocr.predict(str(image_path))
        lines = []

        for page in result:
            if hasattr(page, "get"):
                texts = page.get("rec_texts", [])
                for text in texts:
                    cleaned = str(text).strip()
                    if cleaned:
                        lines.append(cleaned)

        return "\n".join(lines).strip()
    except Exception as e:
        print(f"[ERROR] PaddleOCR failed on {image_path}: {e}")
        return ""


def extract_text_tesseract(image_path, group_name=None):
    image = cv2.imread(str(image_path))
    if image is None:
        return ""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if group_name == "handwritten":
        processed = cv2.bilateralFilter(gray, 9, 75, 75)
        config = "--psm 13"
        return pytesseract.image_to_string(processed, lang="eng", config=config)

    resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    config = "--psm 6"
    return pytesseract.image_to_string(resized, lang="eng", config=config)



from pathlib import Path  # make sure this import is at the top with others


def extract_text_hybrid(image_path, group_name=None):
    if group_name in ["handwritten", "receipts"]:
        return extract_text_tesseract(image_path, group_name=group_name)

    if group_name == "printed":
        image = cv2.imread(str(image_path))
        if image is None:
            return extract_text_paddle(image_path)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_2x = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_2x = clahe.apply(gray_2x)

        tmp_dir = Path("outputs/tmp/printed_paddle")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / (Path(image_path).stem + "_gray2x_clahe.png")
        cv2.imwrite(str(tmp_path), clahe_2x)

        return extract_text_paddle(tmp_path)

    return extract_text_paddle(image_path)

