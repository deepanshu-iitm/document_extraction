## Document Extraction Pipeline

This project implements and evaluates a robust OCR pipeline over multiple document types:

- Degraded document scans
- Dense document pages
- Handwritten text
- Printed documents
- Receipts and invoices
- Scene text images

It compares different OCR engines (Tesseract, PaddleOCR, EasyOCR) and uses a **hybrid routing strategy** that selects the best engine per dataset group, evaluated with the assignment’s CER/WER/combined accuracy metric.

---

### Environment Setup

1. **Create and activate a virtual environment**

```bash
python -m venv venv
venv\Scripts\activate  # on Windows
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

This installs:

- `opencv-python`, `pillow`
- `pytesseract`
- `paddleocr`, `paddlepaddle`
- `jiwer`, `pandas`, `tqdm`

Make sure Tesseract OCR is installed and available on your system path.

---

### Final Pipeline (Hybrid) – How to Run

The **final pipeline** for reporting is the **hybrid OCR pipeline**.

It uses:

- **Tesseract** for:
  - `handwritten`
  - `receipts`
- **PaddleOCR** for:
  - `degraded`
  - `dense_text`
  - `printed`
  - `scene_text`

with small group-specific preprocessing as described below.

#### Single entrypoint

From the project root:

```bash
python run_pipeline.py
```

This will:

1. Run `src/run_hybrid_baseline.py` over all six groups.
2. Save predictions to:

   ```text
   outputs/predictions/hybrid_baseline/<group>/<image>.txt
   ```

3. Save metrics to:

   ```text
   outputs/metrics/hybrid_baseline_detailed_results.csv
   outputs/metrics/hybrid_baseline_group_summary.csv
   outputs/metrics/hybrid_baseline_overall_summary.txt
   ```

4. Print the final **group-wise** and **overall** accuracy to the console.

---

### Evaluation Metrics

All metrics follow the assignment’s provided code and are implemented in `src/evaluate.py`:

- **normalize_text**:
  - Lowercases.
  - Removes non-alphanumeric characters.
  - Collapses whitespace.
- **CER (Character Error Rate)** using `jiwer.cer`.
- **WER (Word Error Rate)** using `jiwer.wer`.
- **Combined Accuracy**:

\[
\text{Accuracy} = \left(1 - \frac{\text{CER} + \text{WER}}{2}\right) \times 100
\]

These functions are used consistently by all pipelines.

---

### Pipelines and Routing Strategy

#### Baseline pipelines

You can also run the pure baselines if needed:

- **Tesseract baseline**:

  ```bash
  python src/run_tesseract_baseline.py
  ```

- **PaddleOCR baseline**:

  ```bash
  python src/run_paddle_baseline.py
  ```

Each baseline:

- Iterates through all groups.
- Runs the chosen OCR engine without routing.
- Writes `*_detailed_results.csv`, `*_group_summary.csv`, and `*_overall_summary.txt` under `outputs/metrics/`.

#### Hybrid pipeline (final)

The hybrid OCR logic lives in `src/ocr_utils.py`:

- **Handwritten (`handwritten`)** – Tesseract:
  - Read with OpenCV, convert to grayscale.
  - Apply bilateral filter to reduce noise.
  - Use Tesseract with `--psm 13` (single text line).
- **Receipts (`receipts`)** – Tesseract:
  - Read with OpenCV, grayscale.
  - 2× resize (`INTER_CUBIC`) to improve small text readability.
  - Use Tesseract with `--psm 6` (single uniform block of text).
- **Degraded (`degraded`)** – PaddleOCR:
  - Use PaddleOCR directly on the raw image path.
- **Dense text (`dense_text`)** – PaddleOCR:
  - Use PaddleOCR directly on the raw image.
- **Scene text (`scene_text`)** – PaddleOCR:
  - Use PaddleOCR directly on the raw image.
  - Additional scaling/preprocessing was tested but did not improve group-level accuracy and was not kept.
- **Printed (`printed`)** – PaddleOCR with preprocessing:
  - Read with OpenCV, grayscale.
  - 2× resize (`INTER_CUBIC`).
  - Apply CLAHE (local contrast enhancement).
  - Save to a temp PNG and pass that to PaddleOCR.
  - This `gray_2x_clahe` variant slightly improves mean printed accuracy over raw/gray/gray_2x.

The hybrid pipeline was chosen after comparing **per-group mean accuracies** of:

- `run_tesseract_baseline.py`
- `run_paddle_baseline.py`
- `run_hybrid_baseline.py`

summarized in `outputs/metrics/pipeline_group_comparison.csv`.

---

### Final Results (Hybrid Pipeline)

Using the hybrid pipeline with the current configuration, the group-wise mean metrics are:

- **Degraded**:
  - CER: 0.6014
  - WER: 0.7328
  - Accuracy: **33.29**
- **Dense text**:
  - CER: 0.2923
  - WER: 0.4467
  - Accuracy: **63.05**
- **Handwritten**:
  - CER: 0.3904
  - WER: 0.8360
  - Accuracy: **38.68**
- **Printed**:
  - CER: 0.4117
  - WER: 0.5720
  - Accuracy: **50.82**
- **Receipts**:
  - CER: 0.0895
  - WER: 0.1625
  - Accuracy: **87.40**
- **Scene text**:
  - CER: 0.1778
  - WER: 0.5000
  - Accuracy: **66.11**

**Overall across all 36 images**:

- overall_cer: **0.3272**
- overall_wer: **0.5417**
- overall_accuracy: **56.56**

(These values are taken from `outputs/metrics/hybrid_baseline_group_summary.csv` and `hybrid_baseline_overall_summary.txt` produced by the latest run.)

---

### Experiments and Design Decisions

Several experiment scripts under `src/exp_*.py` were used to explore improvements:

- **Handwritten** (`exp_handwritten_tesseract_preprocessing.py`, `exp_handwritten_easyocr_baseline.py`):
  - Swept Tesseract preprocessing (resize, median, bilateral, threshold) and PSM modes.
  - Evaluated EasyOCR on the handwritten group.
  - Result: Tesseract with the current handwritten settings outperformed PaddleOCR and EasyOCR on mean accuracy.

- **Degraded** (`exp_degraded_tesseract_preprocessing.py`, `exp_degraded_paddle_preprocessing.py`):
  - Tried multiple preprocessing variants (resizing, denoising, thresholding) for both Tesseract and PaddleOCR.
  - Some heavy preprocessing improved a few images but hurt others; group-level mean accuracy did not improve over the current Paddle baseline.

- **Printed** (`exp_printed_paddle_preprocessing.py`):
  - Compared PaddleOCR on:
    - raw,
    - gray,
    - gray_2x,
    - gray_2x + bilateral,
    - gray_2x + CLAHE.
  - `gray_2x_clahe` gave the best mean accuracy and was adopted in the hybrid pipeline for printed.

- **Scene text** (`exp_scene_text_tesseract_preprocessing.py`, `test_paddleocr_samples.py`):
  - Explored Tesseract with multiple preprocessing variants and PSM modes.
  - Tested 2× scaling and other tricks for PaddleOCR.
  - While Tesseract could sometimes match or beat Paddle on individual images, no single global Tesseract config outperformed PaddleOCR overall, so `scene_text` remains on Paddle.

These experiments are **not** part of the core pipeline, but they justify the final design choices.

---

### Summary

- The project implements:
  - Tesseract and PaddleOCR baselines.
  - A hybrid pipeline that routes groups to the best-performing engine with light, group-specific preprocessing.
- Evaluation strictly follows the assignment’s CER/WER/Accuracy definitions.
- Final results and routing decisions are supported by systematic per-group experiments.
- To reproduce the final metrics and outputs, run:

```bash
python run_pipeline.py
```