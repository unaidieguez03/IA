from utils import image_resizer
from utils import image_grayscaler
from utils import image_normalization
from utils import load_image
import numpy as np
def preprocess(path:str) -> np.ndarray:
    """
    Preprocess a DICOM image through a series of transformations.
    
    Args:
        path (str): File path to the input DICOM image
    
    Returns:
        np.ndarray: Preprocessed image after resizing, grayscale conversion, and normalization
        Returns None if any step in the preprocessing fails
    """
    image = load_image.load_dicom_image(path)
    image = image_resizer.resize_image(image, 512)
    image = image_grayscaler.grayscaler(image)
    image = image_normalization.normalize_image(image)
    return image