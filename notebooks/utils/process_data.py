import cv2
import random
from utils import image_resizer
from utils import image_grayscaler
from utils import image_normalization
from utils import load_image
import numpy as np
AUGMENTATION_PERCENTAGE = 0.5
def augment_from_image(image: np.ndarray, random_state: int | None = None):
    random.seed(random_state)
    rotation_matrix = cv2.getRotationMatrix2D((512 // 2, 512 // 2), random.randint(30, 330), 1.0)
    techniques = [
        lambda img: cv2.flip(img, 1),   # Mirror
        lambda img: cv2.flip(img, 0),   # Upside down
        lambda img: cv2.flip(img, -1),  # Mirror and Upside down
        lambda img: cv2.warpAffine(img, rotation_matrix, (512, 512)) # Rotate
    ]
    technique = random.choice(techniques)
    return technique(image)
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

def preprocess_and_augment(path:str) -> np.ndarray:
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
    image = augment_from_image(image)
    image = image_grayscaler.grayscaler(image)
    image = image_normalization.normalize_image(image)
    return image