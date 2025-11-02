"""
MaxiCode Reader
Reads MaxiCode 2D barcodes from images
MaxiCode is commonly used by UPS for package tracking
"""

# Install required libraries:
# pip install zxing-cpp
# pip install Pillow
# pip install opencv-python
# pip install pyzbar (limited MaxiCode support)

import cv2
import numpy as np
from PIL import Image

def read_maxicode_zxing(image_path):
    """
    Read MaxiCode using ZXing-C++ library (RECOMMENDED)
    ZXing has the best MaxiCode support
    
    Args:
        image_path: Path to the image file
    
    Returns:
        List of decoded MaxiCode data
    """
    try:
        import zxingcpp
        
        # Read image
        img = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Try to read with ZXing
        # MaxiCode specific format
        results = zxingcpp.read_barcodes(gray, formats=zxingcpp.BarcodeFormat.MaxiCode)
        
        if results:
            print(f"✓ Found {len(results)} MaxiCode(s) with ZXing:\n")
            decoded_data = []
            
            for i, result in enumerate(results, 1):
                print(f"MaxiCode {i}:")
                print(f"  Format: {result.format}")
                print(f"  Content Type: {result.content_type}")
                print(f"  Text: {result.text}")
                
                # MaxiCode structured data
                if hasattr(result, 'structured_append'):
                    print(f"  Structured Append: {result.structured_append}")
                
                print(f"  Position: {result.position}")
                print(f"  Orientation: {result.orientation}")
                print()
                
                decoded_data.append({
                    'format': str(result.format),
                    'text': result.text,
                    'content_type': str(result.content_type),
                    'position': str(result.position)
                })
            
            return decoded_data
        else:
            print("✗ No MaxiCode found with ZXing.")
            return []
            
    except ImportError:
        print("ERROR: ZXing-CPP not installed.")
        print("Install with: pip install zxing-cpp")
        print("\nZXing-CPP is REQUIRED for MaxiCode reading!")
        return []
    except Exception as e:
        print(f"Error reading MaxiCode with ZXing: {e}")
        return []


def read_maxicode_pyzbar(image_path):
    """
    Attempt to read MaxiCode using pyzbar (limited support)
    Note: pyzbar has limited MaxiCode support
    """
    try:
        from pyzbar.pyzbar import decode
        
        # Load the image
        img = Image.open(image_path)
        
        # Decode all barcodes
        results = decode(img)
        
        # Filter for MaxiCode
        maxicode_results = [r for r in results if r.type == 'MAXICODE']
        
        if maxicode_results:
            print(f"✓ Found {len(maxicode_results)} MaxiCode(s) with pyzbar:\n")
            decoded_data = []
            
            for i, result in enumerate(maxicode_results, 1):
                print(f"MaxiCode {i}:")
                print(f"  Type: {result.type}")
                
                try:
                    data = result.data.decode('utf-8')
                except:
                    data = result.data.decode('latin-1')
                
                print(f"  Data: {data}")
                print(f"  Position: {result.rect}")
                print()
                
                decoded_data.append({
                    'type': result.type,
                    'data': data,
                    'position': result.rect
                })
            
            return decoded_data
        else:
            print("✗ No MaxiCode found with pyzbar.")
            return []
            
    except ImportError:
        print("pyzbar not installed. Install with: pip install pyzbar")
        return []
    except Exception as e:
        print(f"Error reading MaxiCode with pyzbar: {e}")
        return []


def read_maxicode_advanced(image_path):
    """
    Advanced MaxiCode reading with multiple preprocessing techniques
    Uses ZXing-C++ with various image enhancements
    """
    try:
        import zxingcpp
        
        img = cv2.imread(image_path)
        
        # Store all unique results
        all_results = {}
        
        print("Trying multiple preprocessing techniques with ZXing...\n")
        
        # Technique 1: Original grayscale
        print("1. Original grayscale...")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        results = zxingcpp.read_barcodes(gray, formats=zxingcpp.BarcodeFormat.MaxiCode)
        if results:
            print("   ✓ Success!")
            for r in results:
                all_results[r.text] = r
        else:
            print("   ✗ No MaxiCode found")
        
        # Technique 2: Binary threshold
        print("2. Binary threshold...")
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        results = zxingcpp.read_barcodes(binary, formats=zxingcpp.BarcodeFormat.MaxiCode)
        if results:
            print("   ✓ Success!")
            for r in results:
                all_results[r.text] = r
        else:
            print("   ✗ No MaxiCode found")
        
        # Technique 3: Otsu's thresholding
        print("3. Otsu's thresholding...")
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results = zxingcpp.read_barcodes(otsu, formats=zxingcpp.BarcodeFormat.MaxiCode)
        if results:
            print("   ✓ Success!")
            for r in results:
                all_results[r.text] = r
        else:
            print("   ✗ No MaxiCode found")
        
        # Technique 4: Adaptive thresholding
        print("4. Adaptive thresholding...")
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        results = zxingcpp.read_barcodes(adaptive, formats=zxingcpp.BarcodeFormat.MaxiCode)
        if results:
            print("   ✓ Success!")
            for r in results:
                all_results[r.text] = r
        else:
            print("   ✗ No MaxiCode found")
        
        # Technique 5: Contrast enhancement (CLAHE)
        print("5. Contrast enhancement (CLAHE)...")
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        results = zxingcpp.read_barcodes(enhanced, formats=zxingcpp.BarcodeFormat.MaxiCode)
        if results:
            print("   ✓ Success!")
            for r in results:
                all_results[r.text] = r
        else:
            print("   ✗ No MaxiCode found")
        
        # Technique 6: Inverted image
        print("6. Inverted image...")
        inverted = cv2.bitwise_not(gray)
        results = zxingcpp.read_barcodes(inverted, formats=zxingcpp.BarcodeFormat.MaxiCode)
        if results:
            print("   ✓ Success!")
            for r in results:
                all_results[r.text] = r
        else:
            print("   ✗ No MaxiCode found")
        
        # Technique 7: Gaussian blur + threshold
        print("7. Gaussian blur + threshold...")
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results = zxingcpp.read_barcodes(thresh, formats=zxingcpp.BarcodeFormat.MaxiCode)
        if results:
            print("   ✓ Success!")
            for r in results:
                all_results[r.text] = r
        else:
            print("   ✗ No MaxiCode found")
        
        # Technique 8: Morphological closing
        print("8. Morphological closing...")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        results = zxingcpp.read_barcodes(morph, formats=zxingcpp.BarcodeFormat.MaxiCode)
        if results:
            print("   ✓ Success!")
            for r in results:
                all_results[r.text] = r
        else:
            print("   ✗ No MaxiCode found")
        
        # Display final results
        if all_results:
            print(f"\n{'='*60}")
            print(f"FINAL RESULTS: Found {len(all_results)} unique MaxiCode(s)")
            print('='*60)
            
            decoded_data = []
            for i, (text, result) in enumerate(all_results.items(), 1):
                print(f"\nMaxiCode {i}:")
                print(f"  Format: {result.format}")
                print(f"  Text: {text}")
                print(f"  Content Type: {result.content_type}")
                print(f"  Position: {result.position}")
                
                decoded_data.append({
                    'format': str(result.format),
                    'text': text,
                    'content_type': str(result.content_type)
                })
            
            return decoded_data
        else:
            print("\n✗ No MaxiCode detected with any technique.")
            return []
            
    except ImportError:
        print("ERROR: ZXing-CPP is required for MaxiCode reading!")
        print("Install with: pip install zxing-cpp")
        return []
    except Exception as e:
        print(f"Error: {e}")
        return []


def read_maxicode_from_webcam():
    """
    Read MaxiCode from webcam in real-time
    Press 'q' to quit
    Requires ZXing-C++
    """
    try:
        import zxingcpp
        
        cap = cv2.VideoCapture(0)
        
        print("Starting webcam MaxiCode reader...")
        print("Press 'q' to quit\n")
        
        last_data = None
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Convert to grayscale for better detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Decode MaxiCode
            results = zxingcpp.read_barcodes(gray, formats=zxingcpp.BarcodeFormat.MaxiCode)
            
            # Draw results on frame
            for result in results:
                # Get position
                pos = result.position
                
                # Draw bounding box
                pts = [(int(pos.top_left.x), int(pos.top_left.y)),
                       (int(pos.top_right.x), int(pos.top_right.y)),
                       (int(pos.bottom_right.x), int(pos.bottom_right.y)),
                       (int(pos.bottom_left.x), int(pos.bottom_left.y))]
                pts = np.array(pts, dtype=np.int32)
                cv2.polylines(frame, [pts], True, (0, 255, 0), 3)
                
                # Display code data
                text = result.text
                
                # Only print if it's new data
                if text != last_data:
                    print(f"Detected MaxiCode: {text}")
                    last_data = text
                
                # Display text on frame
                cv2.putText(frame, f"MaxiCode: {text[:30]}", 
                           (int(pos.top_left.x), int(pos.top_left.y) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('MaxiCode Reader (Press Q to quit)', frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
    except ImportError:
        print("ERROR: ZXing-CPP is required for webcam MaxiCode reading!")
        print("Install with: pip install zxing-cpp")
    except Exception as e:
        print(f"Error: {e}")


def save_maxicode_data(data, output_file="maxicode_data.txt"):
    """
    Save decoded MaxiCode data to a file
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, item in enumerate(data, 1):
                f.write(f"MaxiCode {i}:\n")
                f.write(f"  Format: {item.get('format', 'N/A')}\n")
                f.write(f"  Text: {item.get('text', item.get('data', 'N/A'))}\n")
                if 'content_type' in item:
                    f.write(f"  Content Type: {item['content_type']}\n")
                f.write("\n")
        print(f"\n✓ Data saved to {output_file}")
    except Exception as e:
        print(f"Error saving data: {e}")


# Example usage
if __name__ == "__main__":
    import sys
    
    # Replace with your image path
    image_path = "maxicode_image.jpg"
    
    print("="*60)
    print("MAXICODE READER")
    print("="*60)
    print("\nMaxiCode is primarily used by UPS for package tracking")
    print("ZXing-C++ library is REQUIRED for MaxiCode support\n")
    print("="*60 + "\n")
    
    # Try ZXing method (recommended)
    print("Method 1: Using ZXing-C++ (RECOMMENDED)...\n")
    data1 = read_maxicode_zxing(image_path)
    
    # Try pyzbar method (limited support)
    print("\n" + "="*60)
    print("Method 2: Using pyzbar (limited support)...\n")
    data2 = read_maxicode_pyzbar(image_path)
    
    # If both fail, try advanced preprocessing
    if not data1 and not data2:
        print("\n" + "="*60)
        print("Method 3: Advanced preprocessing with ZXing...\n")
        data3 = read_maxicode_advanced(image_path)
    else:
        data3 = []
    
    # Combine all results
    all_data = data1 + data2 + data3
    
    # Save results if found
    if all_data:
        save_maxicode_data(all_data)
        
        # Print final decoded data
        print("\n" + "="*60)
        print("FINAL DECODED DATA")
        print("="*60)
        for item in all_data:
            print(f"\n{item.get('text', item.get('data', 'N/A'))}")
    else:
        print("\n" + "="*60)
        print("No MaxiCode could be decoded.")
        print("="*60)
        print("\nTroubleshooting tips:")
        print("1. Install ZXing-C++: pip install zxing-cpp")
        print("2. Ensure the MaxiCode image is clear and complete")
        print("3. MaxiCode has a distinctive bullseye pattern in center")
        print("4. Try capturing at different angles and lighting")
        print("5. Make sure all hexagonal dots are visible")
    
    # Uncomment to use webcam reader
    # print("\n\nStarting webcam mode...")
    # read_maxicode_from_webcam()
