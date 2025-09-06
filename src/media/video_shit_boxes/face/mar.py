
import numpy as np

def calculate_mar(landmarks: np.ndarray) -> float:
    """
    Calculate the Mouth Aspect Ratio (MAR) from facial landmarks.
    
    Args:
        landmarks: Array of 68 facial landmark points
        
    Returns:
        Mouth aspect ratio value
    """
    # Extract mouth landmarks (points 48-67)
    mouth = landmarks[48:68]
    
    # Calculate vertical distances
    A = np.linalg.norm(mouth[2] - mouth[10])  # 51-59
    B = np.linalg.norm(mouth[4] - mouth[8])   # 53-57
    
    # Calculate horizontal distance
    C = np.linalg.norm(mouth[0] - mouth[6])   # 48-54
    
    # Calculate MAR
    if C == 0:
        return 0.0
    
    return float((A + B) / (2.0 * C))