"""
DataMatrix Code Reader
Reads DataMatrix barcodes from images using pylibdmtx
"""

# First, install required libraries:
# pip install pylibdmtx
# pip install Pillow

from pylibdmtx.pylibdmtx import decode
from PIL import Image

def read_datamatrix(image_path):
    """
    Read DataMatrix code from an image file
    
    Args:
        image_path: Path to the image file
    
    Returns:
        List of decoded data from DataMatrix codes found in the image
    """
    try:
        # Load the image
        img = Image.open(image_path)
        
        # Decode DataMatrix codes
        results = decode(img)
        
        if results:
            print(f"Found {len(results)} DataMatrix code(s):\n")
            decoded_data = []
            
            for i, result in enumerate(results, 1):
                data = result.data.decode('utf-8')
                decoded_data.append(data)
                print(f"Code {i}:")
                print(f"  Data: {data}")
                print(f"  Position: {result.rect}")
                print()
            
            return decoded_data
        else:
            print("No DataMatrix code found in the image.")
            return []
            
    except Exception as e:
        print(f"Error reading DataMatrix code: {e}")
        return []


def read_datamatrix_from_opencv(image_path):
    """
    Alternative method using OpenCV for image preprocessing
    Useful if the image quality is poor
    """
    import cv2
    
    try:
        # Read image with OpenCV
        img = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to improve contrast
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Convert to PIL Image for pylibdmtx
        pil_img = Image.fromarray(thresh)
        
        # Decode DataMatrix codes
        results = decode(pil_img)
        
        if results:
            print(f"Found {len(results)} DataMatrix code(s):\n")
            decoded_data = []
            
            for i, result in enumerate(results, 1):
                data = result.data.decode('utf-8')
                decoded_data.append(data)
                print(f"Code {i}: {data}")
            
            return decoded_data
        else:
            print("No DataMatrix code found in the image.")
            return []
            
    except Exception as e:
        print(f"Error reading DataMatrix code: {e}")
        return []


# Example usage
if __name__ == "__main__":
    # Replace with your image path
    image_path = "M:\\unnamed (1).jpg"
    
    print("Method 1: Direct reading")
    print("-" * 50)
    data = read_datamatrix(image_path)
    
    print("\n" + "="*50 + "\n")
    
    print("Method 2: With OpenCV preprocessing")
    print("-" * 50)
    data = read_datamatrix_from_opencv(image_path)
