"""
metrics.py
==========
Lightweight, vectorized image quality and change detection metrics for proxy frames.
Computes Laplacian sharpness, luminance balance, RMS contrast, inter-frame motion,
and structural/histogram similarity without heavy deep learning overhead.
"""

from typing import Dict, Tuple
import cv2
import numpy as np


def compute_sharpness(gray_frame: np.ndarray) -> float:
    """
    Computes sharpness using the Variance of Laplacian (Var(∇²I)).
    Higher values indicate crisper edges and high-frequency details.
    
    Args:
        gray_frame: Single-channel 8-bit proxy grayscale image.
        
    Returns:
        float: Laplacian variance score (e.g. 50.0 - 2000.0+).
    """
    laplacian = cv2.Laplacian(gray_frame, cv2.CV_64F, ksize=3)
    variance = float(laplacian.var())
    return variance


def compute_brightness(gray_frame: np.ndarray) -> Tuple[float, float]:
    """
    Computes mean luminance and a normalized brightness quality score in [0.0, 1.0].
    Penalizes extreme underexposure (< 30) or extreme overexposure (> 225).
    
    Args:
        gray_frame: Single-channel 8-bit proxy grayscale image.
        
    Returns:
        Tuple[float, float]: (raw_mean_brightness [0-255], normalized_score [0.0-1.0])
    """
    mean_val = float(np.mean(gray_frame))
    # Optimal target luminance is around 128. Gaussian-like decay towards 0 and 255
    # Score is 1.0 at 128, and drops gracefully to 0 at extremes
    normalized_score = max(0.0, 1.0 - abs(mean_val - 128.0) / 128.0)
    return round(mean_val, 2), round(normalized_score, 4)


def compute_contrast(gray_frame: np.ndarray) -> Tuple[float, float]:
    """
    Computes RMS (Root Mean Square) contrast and a normalized contrast score in [0.0, 1.0].
    RMS contrast is defined as the standard deviation of pixel intensities.
    
    Args:
        gray_frame: Single-channel 8-bit proxy grayscale image.
        
    Returns:
        Tuple[float, float]: (rms_contrast [0-128], normalized_score [0.0-1.0])
    """
    rms_contrast = float(np.std(gray_frame))
    # Normalized against max practical standard deviation (~64 for natural outdoor scenes)
    normalized_score = min(1.0, rms_contrast / 64.0)
    return round(rms_contrast, 2), round(normalized_score, 4)


def compute_motion(
    gray_curr: np.ndarray,
    gray_prev: np.ndarray,
    motion_thresh: int = 25
) -> float:
    """
    Computes inter-frame motion magnitude using thresholded frame differencing.
    Reflects the percentage of active scene change between successive candidate frames.
    
    Args:
        gray_curr: Current proxy grayscale image.
        gray_prev: Preceding candidate proxy grayscale image.
        motion_thresh: Pixel difference threshold (0-255) for active motion.
        
    Returns:
        float: Motion score in [0.0, 1.0], representing normalized motion/change area.
    """
    if gray_prev is None or gray_curr.shape != gray_prev.shape:
        return 1.0  # First candidate frame has maximal new information
        
    diff = cv2.absdiff(gray_curr, gray_prev)
    _, thresh = cv2.threshold(diff, motion_thresh, 255, cv2.THRESH_BINARY)
    
    # Ratio of pixels with significant change
    motion_fraction = float(np.count_nonzero(thresh)) / float(thresh.size)
    # Mean intensity change as secondary factor
    mean_delta = float(np.mean(diff)) / 255.0
    
    # Combined motion score
    motion_score = min(1.0, 0.7 * motion_fraction * 5.0 + 0.3 * mean_delta * 3.0)
    return round(motion_score, 4)


def compute_similarity(
    bgr_curr: np.ndarray,
    bgr_ref: np.ndarray
) -> float:
    """
    Computes similarity between the current frame and the last selected frame.
    Uses joint 3D/2D color histogram correlation in HSV space + fast normalized MSE.
    Returns 1.0 if frames are identical, 0.0 if completely distinct scenes.
    
    Args:
        bgr_curr: Current candidate BGR proxy image.
        bgr_ref: Last selected BGR proxy image.
        
    Returns:
        float: Similarity score in [0.0, 1.0].
    """
    if bgr_ref is None:
        return 0.0  # No previous selected frame, perfectly novel
        
    # Convert to HSV for illumination-robust color comparison
    hsv_curr = cv2.cvtColor(bgr_curr, cv2.COLOR_BGR2HSV)
    hsv_ref = cv2.cvtColor(bgr_ref, cv2.COLOR_BGR2HSV)
    
    # 2D H-S histogram
    hist_curr = cv2.calcHist([hsv_curr], [0, 1], None, [30, 32], [0, 180, 0, 256])
    hist_ref = cv2.calcHist([hsv_ref], [0, 1], None, [30, 32], [0, 180, 0, 256])
    
    cv2.normalize(hist_curr, hist_curr, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist_ref, hist_ref, 0, 1, cv2.NORM_MINMAX)
    
    # Histogram Correlation [-1.0, 1.0] -> map to [0.0, 1.0]
    hist_corr = max(0.0, float(cv2.compareHist(hist_curr, hist_ref, cv2.HISTCMP_CORREL)))
    
    # Fast Normalized Pixel Difference
    diff = cv2.absdiff(bgr_curr, bgr_ref)
    pixel_similarity = max(0.0, 1.0 - (float(np.mean(diff)) / 128.0))
    
    combined_similarity = 0.6 * hist_corr + 0.4 * pixel_similarity
    return round(float(np.clip(combined_similarity, 0.0, 1.0)), 4)


def compute_candidate_metrics(
    proxy_bgr_curr: np.ndarray,
    proxy_gray_curr: np.ndarray,
    proxy_gray_prev: np.ndarray,
    proxy_bgr_last_selected: np.ndarray,
    blur_threshold: float = 100.0,
    quality_weights: Tuple[float, float, float] = (0.5, 0.25, 0.25)
) -> Dict[str, float]:
    """
    Evaluates all image quality and temporal change metrics for a candidate frame.
    
    Args:
        proxy_bgr_curr: Current BGR proxy frame.
        proxy_gray_curr: Current Grayscale proxy frame.
        proxy_gray_prev: Preceding candidate Grayscale proxy frame.
        proxy_bgr_last_selected: Grayscale or BGR proxy of the last selected frame.
        blur_threshold: Laplacian variance cutoff for blur determination.
        quality_weights: Weights for (sharpness_norm, brightness_norm, contrast_norm).
        
    Returns:
        Dict of computed scores.
    """
    sharpness = compute_sharpness(proxy_gray_curr)
    mean_bright, bright_score = compute_brightness(proxy_gray_curr)
    rms_contrast, contrast_score = compute_contrast(proxy_gray_curr)
    motion_score = compute_motion(proxy_gray_curr, proxy_gray_prev)
    similarity_score = compute_similarity(proxy_bgr_curr, proxy_bgr_last_selected)
    
    # Normalized sharpness score in [0.0, 1.0] (saturating at 3x blur threshold)
    sharpness_norm = min(1.0, sharpness / max(blur_threshold * 3.0, 1.0))
    
    # Composite Quality Score [0.0, 1.0]
    w_s, w_b, w_c = quality_weights
    composite_quality = (
        w_s * sharpness_norm +
        w_b * bright_score +
        w_c * contrast_score
    )
    
    # Comprehensive selection score: quality * novelty * motion factor
    # Novelty is (1.0 - similarity_score)
    novelty = 1.0 - similarity_score
    selection_score = 0.4 * composite_quality + 0.35 * novelty + 0.25 * motion_score
    
    return {
        "sharpness": round(sharpness, 2),
        "sharpness_norm": round(sharpness_norm, 4),
        "brightness": mean_bright,
        "brightness_score": bright_score,
        "contrast": rms_contrast,
        "contrast_score": contrast_score,
        "motion_score": motion_score,
        "similarity_score": similarity_score,
        "quality_score": round(composite_quality, 4),
        "selection_score": round(selection_score, 4),
        "is_blurry": sharpness < blur_threshold
    }
