import cv2
import numpy as np
import polars as pl

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