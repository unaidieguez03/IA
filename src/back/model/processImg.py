import cv2
import polars as pl
import numpy as np

def grayscaler(image: np.ndarray) -> np.ndarray:
    """
    Convert an image to grayscale, handling different input image formats.
    
    Args:
        image (np.ndarray): Input image to convert to grayscale
    
    Returns:
        np.ndarray: Grayscale image, or None if conversion fails
    """
    if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        return image
    if len(image.shape) == 3 and image.shape[2] == 3:
        grayscale_image: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        grayscale_image: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        print(f"Unsupported image shape: {image.shape}")
        return None
    return grayscale_image
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
def resize_image(image, size) -> np.ndarray:
    """
    Resize an image to a specified size using area interpolation.
    
    Args:
        image (np.ndarray): Input image to be resized
        size (int): Target width and height (square resizing)
    
    Returns:
        np.ndarray: Resized image, or None if resizing fails
    """
    try:
        resized_image: np.ndarray = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        return resized_image
    except Exception as e:
        print(f"Error pene DICOM image {e}")
        return None