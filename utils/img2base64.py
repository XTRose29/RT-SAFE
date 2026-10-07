import base64
import io
from PIL import Image
import numpy as np
import cv2

def pil_to_base64(pil_image, format='PNG', quality=95):
    """
    将 PIL Image 转换为 base64 字符串
    
    Args:
        pil_image: PIL Image 对象
        format: 图像格式 ('JPEG', 'PNG', etc.)
        quality: JPEG 质量 (1-100)，仅对 JPEG 有效
    
    Returns:
        base64_string: base64 编码的字符串
    """
    buffer = io.BytesIO()
    
    if format.upper() == 'JPEG':
        pil_image.save(buffer, format=format, quality=quality)
    else:
        pil_image.save(buffer, format=format)
    
    image_bytes = buffer.getvalue()
    
    base64_string = base64.b64encode(image_bytes).decode('utf-8')
    
    return base64_string

def np_to_base64(image: np.ndarray) -> str:
        """Convert numpy array image to base64 string.

        Args:
            image (np.ndarray): Image array (1 or 3 channels)

        Returns:
            str: Base64 encoded image string
        """
        # Convert single channel to 3 channels if needed
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Ensure uint8 type
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        # Convert to PIL Image
        pil_image = Image.fromarray(image)

        # Convert to base64
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return img_str