"""
Universal Barcode Reader
Reads various barcode types (UPC, EAN, Code128, Code39, etc.)
"""

# Install required libraries:
# pip install pyzbar
# pip install Pillow
# pip install opencv-python

from pyzbar.pyzbar import decode
from PIL import Image
import cv2
import numpy as np

def read_barcode_simple(image_path):
    """
    Simple barcode reading method
    
    Args:
        image_path: Path to the image file
    
    Returns:
        List of decoded barcode data
    """
    try:
        # Load the image
        img = Image.open(image_path)
        
        # Decode all barcodes
        results = decode(img)
        
        if results:
            print(f"✓ Found {len(results)} barcode(s):\n")
            decoded_data = []
            
            for i, result in enumerate(results, 1):
                print(f"Barcode {i}:")
                print(f"  Type: {result.type}")
                print(f"  Data: {result.data.decode('utf-8')}")
                print(f"  Position: {result.rect}")
                print(f"  Quality: {result.quality}")
                print()
                
                decoded_data.append({
                    'type': result.type,
                    'data': result.data.decode('utf-8'),
                    'position': result.rect,
                    'quality': result.quality
                })
            
            return decoded_data
        else:
            print("✗ No barcode found in the image.")
            return []
            
    except Exception as e:
        print(f"Error reading barcode: {e}")
        return []


def read_barcode_with_preprocessing(image_path):
    """
    Read barcode with image preprocessing
    Handles various image quality issues
    """
    try:
        # Read image with OpenCV
        img = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Decode barcodes
        results = decode(thresh)
        
        if results:
            print(f"✓ Found {len(results)} barcode(s) with preprocessing:\n")
            decoded_data = []
            
            for i, result in enumerate(results, 1):
                print(f"Barcode {i}:")
                print(f"  Type: {result.type}")
                print(f"  Data: {result.data.decode('utf-8')}")
                print()
                
                decoded_data.append({
                    'type': result.type,
                    'data': result.data.decode('utf-8')
                })
            
            return decoded_data
        else:
            print("✗ No barcode found after preprocessing.")
            return []
            
    except Exception as e:
        print(f"Error: {e}")
        return []


def read_barcode_advanced(image_path):
    """
    Advanced barcode reading with multiple preprocessing techniques
    Tries various methods to maximize success rate
    """
    try:
        img = cv2.imread(image_path)
        
        # Store all unique results
        all_results = {}
        
        print("Trying multiple preprocessing techniques...\n")
        
        # Technique 1: Original grayscale
        print("1. Original grayscale...")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        results = decode(gray)
        if results:
            print("   ✓ Success!")
            for r in results:
                all_results[r.data.decode('utf-8')] = r
        else:
            print("   ✗ No barcode found")
        
        # Technique 2: Binary threshold
        print("2. Binary threshold...")
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        results = decode(binary)
        if results:
            print("   ✓ Success!")
            for r in results:
                all_results[r.data.decode('utf-8')] = r
        else:
            print("   ✗ No barcode found")
        
        # Technique 3: Otsu's thresholding
        print("3. Otsu's thresholding...")
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results = decode(otsu)
        if results:
            print("   ✓ Success!")
            for r in results:
                all_results[r.data.decode('utf-8')] = r
        else:
            print("   ✗ No barcode found")
        
        # Technique 4: Contrast enhancement
        print("4. Contrast enhancement...")
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        results = decode(enhanced)
        if results:
            print("   ✓ Success!")
            for r in results:
                all_results[r.data.decode('utf-8')] = r
        else:
            print("   ✗ No barcode found")
        
        # Technique 5: Sharpening
        print("5. Sharpening...")
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        results = decode(sharpened)
        if results:
            print("   ✓ Success!")
            for r in results:
                all_results[r.data.decode('utf-8')] = r
        else:
            print("   ✗ No barcode found")
        
        # Technique 6: Morphological operations
        print("6. Morphological operations...")
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        results = decode(morph)
        if results:
            print("   ✓ Success!")
            for r in results:
                all_results[r.data.decode('utf-8')] = r
        else:
            print("   ✗ No barcode found")
        
        # Display final results
        if all_results:
            print(f"\n{'='*60}")
            print(f"FINAL RESULTS: Found {len(all_results)} unique barcode(s)")
            print('='*60)
            
            decoded_data = []
            for i, (data, result) in enumerate(all_results.items(), 1):
                print(f"\nBarcode {i}:")
                print(f"  Type: {result.type}")
                print(f"  Data: {data}")
                print(f"  Position: {result.rect}")
                
                decoded_data.append({
                    'type': result.type,
                    'data': data,
                    'position': result.rect
                })
            
            return decoded_data
        else:
            print("\n✗ No barcodes detected with any technique.")
            return []
            
    except Exception as e:
        print(f"Error: {e}")
        return []


def read_barcode_from_webcam():
    """
    Read barcode from webcam in real-time
    Press 'q' to quit
    """
    cap = cv2.VideoCapture(0)
    
    print("Starting webcam barcode reader...")
    print("Press 'q' to quit\n")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Decode barcodes
        results = decode(frame)
        
        # Draw results on frame
        for result in results:
            # Get barcode position
            pts = result.polygon
            
            if len(pts) == 4:
                # Draw barcode boundary
                pts = [(point.x, point.y) for point in pts]
                pts = np.array(pts, dtype=np.int32)
                cv2.polylines(frame, [pts], True, (0, 255, 0), 3)
            
            # Display barcode data
            data = result.data.decode('utf-8')
            text = f"{result.type}: {data}"
            cv2.putText(frame, text, (result.rect.left, result.rect.top - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            print(f"Detected: {result.type} - {data}")
        
        # Display frame
        cv2.imshow('Barcode Reader', frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def save_barcode_data(data, output_file="barcode_data.txt"):
    """
    Save decoded barcode data to a file
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, item in enumerate(data, 1):
                f.write(f"Barcode {i}:\n")
                f.write(f"  Type: {item['type']}\n")
                f.write(f"  Data: {item['data']}\n")
                if 'position' in item:
                    f.write(f"  Position: {item['position']}\n")
                f.write("\n")
        print(f"\n✓ Data saved to {output_file}")
    except Exception as e:
        print(f"Error saving data: {e}")


# Example usage
if __name__ == "__main__":
    import sys
    
    # Replace with your image path
    image_path = "M:\\barcode.jpg"
    
    print("="*60)
    print("BARCODE READER")
    print("="*60 + "\n")
    
    # Try simple method first
    print("Attempting simple read...\n")
    data = read_barcode_simple(image_path)
    
    # If simple method fails, try advanced method
    if not data:
        print("\n" + "="*60)
        print("Trying advanced preprocessing...\n")
        data = read_barcode_advanced(image_path)
    
    # Save results if found
    if data:
        save_barcode_data(data)
    
    # Uncomment to use webcam reader
    # read_barcode_from_webcam()
