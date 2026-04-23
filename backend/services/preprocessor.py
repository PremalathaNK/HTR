"""
Image Pre-processing Pipeline for ScriptBridge HTR.

Two pipeline modes controlled by the `mode` parameter:

  mode='easyocr'  (DEFAULT)
    - Resize  → Deskew  → Denoise (mild)  → CLAHE contrast enhance
    - NO binary thresholding — EasyOCR needs real pixel gradients to
      find text regions; binary images collapse those gradients to zero
      and make EasyOCR return empty results.

  mode='trocr'
    - Resize  → Deskew  → Denoise  → Adaptive Threshold  → RGB
    - Binary thresholding improves TrOCR accuracy on handwritten notes
      by giving the encoder a clean, high-contrast patch.

Quality scoring (Laplacian variance) is computed in both modes.
"""

import cv2
import numpy as np
from PIL import Image
import io
from typing import Literal


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV BGR array."""
    return cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR array to PIL Image (RGB)."""
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))


def compute_quality_score(gray: np.ndarray) -> float:
    """
    Compute image sharpness score using Laplacian variance.
    Higher = sharper. Below ~50 usually means blurry.
    """
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(laplacian_var)


def deskew(image: np.ndarray) -> np.ndarray:
    """
    Detect and correct image skew using Hough line transform.
    Works best for ±15 degree tilts.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                             minLineLength=100, maxLineGap=10)

    if lines is None:
        return image  # Can't detect skew, return as-is

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 != 0:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if -20 < angle < 20:
                angles.append(angle)

    if not angles:
        return image

    median_angle = np.median(angles)
    if abs(median_angle) < 0.5:
        return image  # No meaningful skew

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    deskewed = cv2.warpAffine(image, rotation_matrix, (w, h),
                               flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)
    return deskewed


PreprocessMode = Literal["easyocr", "trocr"]


def preprocess_image(pil_image: Image.Image, mode: PreprocessMode = "easyocr") -> dict:
    """
    Pre-processing pipeline with two modes.

    Args:
        pil_image: Input PIL image (any mode)
        mode: 'easyocr' (light, keeps gradients) or 'trocr' (binary threshold)

    Returns:
        dict with keys:
            - processed_image (PIL.Image): Enhanced image ready for OCR
            - quality_score (float): Sharpness score (Laplacian variance)
            - is_blurry (bool): True if quality_score < 50
            - skew_corrected (bool): True if deskew was applied
            - mode (str): Which pipeline was applied
    """
    # ── Step 1: Convert to OpenCV BGR ──────────────────────────────────────
    img_cv = pil_to_cv2(pil_image)

    # ── Step 2: Resize (scale up if too small) ──────────────────────────────
    h, w = img_cv.shape[:2]
    if max(h, w) < 800:
        scale = 800 / max(h, w)
        img_cv = cv2.resize(img_cv, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_CUBIC)

    # ── Step 3: Deskew ──────────────────────────────────────────────────────
    original_for_skew = img_cv.copy()
    img_cv = deskew(img_cv)
    skew_corrected = not np.array_equal(img_cv, original_for_skew)

    # ── Step 4: Grayscale + quality score ───────────────────────────────────
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    quality_score = compute_quality_score(gray)
    is_blurry = quality_score < 50.0

    # ═══════════════════════════════════════════════════════════════════════
    # EASYOCR MODE — light pipeline, preserve gradient information
    # ═══════════════════════════════════════════════════════════════════════
    if mode == "easyocr":
        # Mild denoise (lower h = less smoothing → keeps character edges)
        denoised = cv2.fastNlMeansDenoising(gray, h=7, templateWindowSize=7,
                                             searchWindowSize=15)

        # CLAHE: improve local contrast without destroying gradients
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # Keep as 3-channel RGB (EasyOCR expects colour or grayscale-as-RGB)
        processed_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        processed_pil = Image.fromarray(processed_rgb)

    # ═══════════════════════════════════════════════════════════════════════
    # TROCR MODE — full pipeline with adaptive threshold
    # ═══════════════════════════════════════════════════════════════════════
    else:
        # Stronger denoise for cleaner binary
        denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7,
                                             searchWindowSize=21)

        # Adaptive threshold → binary image (best for TrOCR patch encoder)
        thresholded = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2,
        )

        # Convert back to 3-channel for TrOCR compatibility
        processed_rgb = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB)
        processed_pil = Image.fromarray(processed_rgb)

    return {
        "processed_image": processed_pil,
        "quality_score": round(quality_score, 2),
        "is_blurry": is_blurry,
        "skew_corrected": skew_corrected,
        "mode": mode,
    }


def bytes_to_pil(image_bytes: bytes) -> Image.Image:
    """Convert raw image bytes to PIL Image."""
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")
