import pydicom
import numpy as np

def load_dicom_image(path_to_image) -> np.ndarray:
    """
    Load a DICOM image from the specified file path.
    
    Args:
        path_to_image (str): Full path to the DICOM image file
    
    Returns:
        np.ndarray: Pixel array of the DICOM image, or None if loading fails
    """
    try:
        dicom_data = pydicom.dcmread(path_to_image)
        image: np.ndarray = dicom_data.pixel_array
        return image
    except Exception as e:
        print(f"Error loading DICOM image {path_to_image}: {e}")
        return None