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