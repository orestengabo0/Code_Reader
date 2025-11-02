"""
QR Code Reader
Reads QR codes from images with multiple preprocessing techniques
"""

# Install required libraries:
# pip install pyzbar
# pip install Pillow
# pip install opencv-python
# pip install qrcode[pil]

from pyzbar.pyzbar import decode
from PIL import Image
import cv2
import numpy as np

def read_qrcode_simple(image_path):
    """
    Simple QR code reading method
    
    Args:
        image_path: Path to the image file
    
    Returns:
        List of decoded QR code data
    """
    try:
        # Load the image
        img = Image.open(image_path)
        
        # Decode QR codes
        results = decode(img)
        
        if results:
            print(f"✓ Found {len(results)} QR code(s):\n")
            decoded_data = []
            
            for i, result in enumerate(results, 1):
                print(f"QR Code {i}:")
                print(f"  Type: {result.type}")
                
                # Try to decode as UTF-8
                try:
                    data = result.data.decode('utf-8')
                except:
                    data = result.data.decode('latin-1')
                
                print(f"  Data: {data}")
                print(f"  Position: {result.rect}")
                print(f"  Quality: {result.quality}")
                print()
                
                decoded_data.append({
                    'type': result.type,
                    'data': data,
                    'position': result.rect,
                    'quality': result.quality
                })
            
            return decoded_data
        else:
            print("✗ No QR code found in the image.")
            return []
            
    except Exception as e:
        print(f"Error reading QR code: {e}")
        return []


def read_qrcode_with_preprocessing(image_path):
    """
    Read QR code with image preprocessing
    Handles low-quality, blurry, or damaged QR codes
    """
    try:
        # Read image with OpenCV
        img = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while keeping edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Decode QR codes
        results = decode(thresh)
        
        if results:
            print(f"✓ Found {len(results)} QR code(s) with preprocessing:\n")
            decoded_data = []
            
            for i, result in enumerate(results, 1):
                print(f"QR Code {i}:")
                print(f"  Type: {result.type}")
                
                try:
                    data = result.data.decode('utf-8')
                except:
                    data = result.data.decode('latin-1')
                
                print(f"  Data: {data}")
                print()
                
                decoded_data.append({
                    'type': result.type,
                    'data': data
                })
            
            return decoded_data
        else:
            print("✗ No QR code found after preprocessing.")
            return []
            
    except Exception as e:
        print(f"Error: {e}")
        return []


def read_qrcode_advanced(image_path):
    """
    Advanced QR code reading with multiple preprocessing techniques
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
                try:
                    data = r.data.decode('utf-8')
                except:
                    data = r.data.decode('latin-1')
                all_results[data] = r
        else:
            print("   ✗ No QR code found")
        
        # Technique 2: Otsu's thresholding
        print("2. Otsu's thresholding...")
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results = decode(otsu)
        if results:
            print("   ✓ Success!")
            for r in results:
                try:
                    data = r.data.decode('utf-8')
                except:
                    data = r.data.decode('latin-1')
                all_results[data] = r
        else:
            print("   ✗ No QR code found")
        
        # Technique 3: Adaptive thresholding
        print("3. Adaptive thresholding...")
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        results = decode(adaptive)
        if results:
            print("   ✓ Success!")
            for r in results:
                try:
                    data = r.data.decode('utf-8')
                except:
                    data = r.data.decode('latin-1')
                all_results[data] = r
        else:
            print("   ✗ No QR code found")
        
        # Technique 4: Contrast enhancement (CLAHE)
        print("4. Contrast enhancement (CLAHE)...")
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        results = decode(enhanced)
        if results:
            print("   ✓ Success!")
            for r in results:
                try:
                    data = r.data.decode('utf-8')
                except:
                    data = r.data.decode('latin-1')
                all_results[data] = r
        else:
            print("   ✗ No QR code found")
        
        # Technique 5: Gaussian blur + threshold
        print("5. Gaussian blur + threshold...")
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results = decode(thresh)
        if results:
            print("   ✓ Success!")
            for r in results:
                try:
                    data = r.data.decode('utf-8')
                except:
                    data = r.data.decode('latin-1')
                all_results[data] = r
        else:
            print("   ✗ No QR code found")
        
        # Technique 6: Morphological operations
        print("6. Morphological operations...")
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        results = decode(morph)
        if results:
            print("   ✓ Success!")
            for r in results:
                try:
                    data = r.data.decode('utf-8')
                except:
                    data = r.data.decode('latin-1')
                all_results[data] = r
        else:
            print("   ✗ No QR code found")
        
        # Technique 7: Sharpening
        print("7. Sharpening...")
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        results = decode(sharpened)
        if results:
            print("   ✓ Success!")
            for r in results:
                try:
                    data = r.data.decode('utf-8')
                except:
                    data = r.data.decode('latin-1')
                all_results[data] = r
        else:
            print("   ✗ No QR code found")
        
        # Technique 8: Inverted image (for negative QR codes)
        print("8. Inverted image...")
        inverted = cv2.bitwise_not(gray)
        results = decode(inverted)
        if results:
            print("   ✓ Success!")
            for r in results:
                try:
                    data = r.data.decode('utf-8')
                except:
                    data = r.data.decode('latin-1')
                all_results[data] = r
        else:
            print("   ✗ No QR code found")
        
        # Display final results
        if all_results:
            print(f"\n{'='*60}")
            print(f"FINAL RESULTS: Found {len(all_results)} unique QR code(s)")
            print('='*60)
            
            decoded_data = []
            for i, (data, result) in enumerate(all_results.items(), 1):
                print(f"\nQR Code {i}:")
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
            print("\n✗ No QR codes detected with any technique.")
            return []
            
    except Exception as e:
        print(f"Error: {e}")
        return []


def read_qrcode_from_webcam():
    """
    Read QR code from webcam in real-time
    Press 'q' to quit
    """
    cap = cv2.VideoCapture(0)
    
    print("Starting webcam QR code reader...")
    print("Press 'q' to quit\n")
    
    last_data = None
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Decode QR codes
        results = decode(frame)
        
        # Draw results on frame
        for result in results:
            # Get QR code position
            pts = result.polygon
            
            if len(pts) == 4:
                # Draw QR code boundary
                pts = [(point.x, point.y) for point in pts]
                pts = np.array(pts, dtype=np.int32)
                cv2.polylines(frame, [pts], True, (0, 255, 0), 3)
            
            # Display QR code data
            try:
                data = result.data.decode('utf-8')
            except:
                data = result.data.decode('latin-1')
            
            # Only print if it's new data
            if data != last_data:
                print(f"Detected QR Code: {data}")
                last_data = data
            
            # Display text on frame
            cv2.putText(frame, data[:50], (result.rect.left, result.rect.top - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow('QR Code Reader (Press Q to quit)', frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def save_qrcode_data(data, output_file="qrcode_data.txt"):
    """
    Save decoded QR code data to a file
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, item in enumerate(data, 1):
                f.write(f"QR Code {i}:\n")
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
    image_path = "M:\\qrcode.jpg"
    
    print("="*60)
    print("QR CODE READER")
    print("="*60 + "\n")
    
    # Try simple method first
    print("Attempting simple read...\n")
    data = read_qrcode_simple(image_path)
    
    # If simple method fails, try advanced method
    if not data:
        print("\n" + "="*60)
        print("Trying advanced preprocessing...\n")
        data = read_qrcode_advanced(image_path)
    
    # Save results if found
    if data:
        save_qrcode_data(data)
        
        # Print decoded data
        print("\n" + "="*60)
        print("DECODED DATA")
        print("="*60)
        for item in data:
            print(f"\n{item['data']}")
    
    # Uncomment to use webcam reader
    # print("\n\nStarting webcam mode...")
    # read_qrcode_from_webcam()
