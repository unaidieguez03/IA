import cv2
import numpy as np

def normalize_image(image) -> np.ndarray:
    """
    Normalize an image to the range [0, 1] using min-max normalization.
    
    Args:
        image (np.ndarray): Input image to normalize
    
    Returns:
        np.ndarray: Normalized image in float32 format, or None if normalization fails
    """
    try:
        normalized_image: np.ndarray = cv2.normalize(
            image,
            None,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F
        )
        
        return normalized_image
    
    except Exception as e:
        print(f"Error {e}")
        return None