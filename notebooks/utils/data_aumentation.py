import numpy as np

def mirror_image(image: np.ndarray) -> np.ndarray:
    """
    Flip the image horizontally.
    Args:
        image (np.ndarray): Input image array.
    Returns:
        np.ndarray: Horizontally flipped image.
    """
    return np.fliplr(image)

def upside_down_image(image: np.ndarray) -> np.ndarray:
    """
    Flip the image vertically.
    Args:
        image (np.ndarray): Input image array.
    Returns:
        np.ndarray: Vertically flipped image.
    """
    return np.flipud(image)

def rotate_90_clockwise(image: np.ndarray) -> np.ndarray:
    """
    Rotate the image 90 degrees clockwise.
    Args:
        image (np.ndarray): Input image array.
    Returns:
        np.ndarray: Image rotated 90 degrees clockwise.
    """
    return np.rot90(image, k=-1)

def rotate_90_counterclockwise(image: np.ndarray) -> np.ndarray:
    """
    Rotate the image 90 degrees counterclockwise.
    Args:
        image (np.ndarray): Input image array.
    Returns:
        np.ndarray: Image rotated 90 degrees counterclockwise.
    """
    return np.rot90(image, k=1)
