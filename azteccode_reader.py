"""
Aztec Code Reader
Reads Aztec codes from images with multiple preprocessing techniques
"""

# Install required libraries:
# pip install pyzbar
# pip install Pillow
# pip install opencv-python
# pip install zxing-cpp  # Alternative decoder with better Aztec support

from pyzbar.pyzbar import decode
from PIL import Image
import cv2
import numpy as np

def read_aztec_simple(image_path):
    """
    Simple Aztec code reading method using pyzbar
    
    Args:
        image_path: Path to the image file
    
    Returns:
        List of decoded Aztec code data
    """
    try:
        # Load the image
        img = Image.open(image_path)
        
        # Decode Aztec codes
        results = decode(img)
        
        if results:
            print(f"✓ Found {len(results)} code(s):\n")
            decoded_data = []
            
            for i, result in enumerate(results, 1):
                print(f"Code {i}:")
                print(f"  Type: {result.type}")
                
                # Try to decode as UTF-8
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
            print("✗ No Aztec code found in the image.")
            return []
            
    except Exception as e:
        print(f"Error reading Aztec code: {e}")
        return []


def read_aztec_with_zxing(image_path):
    """
    Read Aztec code using ZXing-C++ library
    ZXing generally has better Aztec code support than pyzbar
    """
    try:
        import zxingcpp
        
        # Read image
        img = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Try to read with ZXing
        results = zxingcpp.read_barcodes(gray)
        
        if results:
            print(f"✓ Found {len(results)} Aztec code(s) with ZXing:\n")
            decoded_data = []
            
            for i, result in enumerate(results, 1):
                print(f"Code {i}:")
                print(f"  Type: {result.format}")
                print(f"  Data: {result.text}")
                print(f"  Position: {result.position}")
                print()
                
                decoded_data.append({
                    'type': str(result.format),
                    'data': result.text,
                    'position': str(result.position)
                })
            
            return decoded_data
        else:
            print("✗ No Aztec code found with ZXing.")
            return []
            
    except ImportError:
        print("ZXing-CPP not installed. Install with: pip install zxing-cpp")
        return []
    except Exception as e:
        print(f"Error reading Aztec code with ZXing: {e}")
        return []


def read_aztec_advanced(image_path):
    """
    Advanced Aztec code reading with multiple preprocessing techniques
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
            print("   ✓ Success with pyzbar!")
            for r in results:
                try:
                    data = r.data.decode('utf-8')
                except:
                    data = r.data.decode('latin-1')
                all_results[data] = ('pyzbar', r)
        else:
            print("   ✗ No code found with pyzbar")
        
        # Try with ZXing on original
        try:
            import zxingcpp
            results_zxing = zxingcpp.read_barcodes(gray)
            if results_zxing:
                print("   ✓ Success with ZXing!")
                for r in results_zxing:
                    all_results[r.text] = ('zxing', r)
            else:
                print("   ✗ No code found with ZXing")
        except:
            pass
        
        # Technique 2: Binary threshold
        print("2. Binary threshold...")
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        results = decode(binary)
        if results:
            print("   ✓ Success!")
            for r in results:
                try:
                    data = r.data.decode('utf-8')
                except:
                    data = r.data.decode('latin-1')
                all_results[data] = ('pyzbar', r)
        else:
            print("   ✗ No code found")
        
        # Technique 3: Otsu's thresholding
        print("3. Otsu's thresholding...")
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results = decode(otsu)
        if results:
            print("   ✓ Success!")
            for r in results:
                try:
                    data = r.data.decode('utf-8')
                except:
                    data = r.data.decode('latin-1')
                all_results[data] = ('pyzbar', r)
        else:
            print("   ✗ No code found")
        
        # Try ZXing on Otsu
        try:
            import zxingcpp
            results_zxing = zxingcpp.read_barcodes(otsu)
            if results_zxing:
                print("   ✓ Success with ZXing on Otsu!")
                for r in results_zxing:
                    all_results[r.text] = ('zxing', r)
        except:
            pass
        
        # Technique 4: Adaptive thresholding
        print("4. Adaptive thresholding...")
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
                all_results[data] = ('pyzbar', r)
        else:
            print("   ✗ No code found")
        
        # Technique 5: Inverted image
        print("5. Inverted image...")
        inverted = cv2.bitwise_not(gray)
        results = decode(inverted)
        if results:
            print("   ✓ Success!")
            for r in results:
                try:
                    data = r.data.decode('utf-8')
                except:
                    data = r.data.decode('latin-1')
                all_results[data] = ('pyzbar', r)
        else:
            print("   ✗ No code found")
        
        # Try ZXing on inverted
        try:
            import zxingcpp
            results_zxing = zxingcpp.read_barcodes(inverted)
            if results_zxing:
                print("   ✓ Success with ZXing on inverted!")
                for r in results_zxing:
                    all_results[r.text] = ('zxing', r)
        except:
            pass
        
        # Technique 6: Morphological operations
        print("6. Morphological operations...")
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        results = decode(morph)
        if results:
            print("   ✓ Success!")
            for r in results:
                try:
                    data = r.data.decode('utf-8')
                except:
                    data = r.data.decode('latin-1')
                all_results[data] = ('pyzbar', r)
        else:
            print("   ✗ No code found")
        
        # Technique 7: Contrast enhancement
        print("7. Contrast enhancement (CLAHE)...")
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
                all_results[data] = ('pyzbar', r)
        else:
            print("   ✗ No code found")
        
        # Display final results
        if all_results:
            print(f"\n{'='*60}")
            print(f"FINAL RESULTS: Found {len(all_results)} unique code(s)")
            print('='*60)
            
            decoded_data = []
            for i, (data, (decoder, result)) in enumerate(all_results.items(), 1):
                print(f"\nCode {i} (decoded with {decoder}):")
                print(f"  Data: {data}")
                
                decoded_data.append({
                    'decoder': decoder,
                    'data': data
                })
            
            return decoded_data
        else:
            print("\n✗ No Aztec codes detected with any technique.")
            return []
            
    except Exception as e:
        print(f"Error: {e}")
        return []


def read_aztec_from_webcam():
    """
    Read Aztec code from webcam in real-time
    Press 'q' to quit
    """
    cap = cv2.VideoCapture(0)
    
    print("Starting webcam Aztec code reader...")
    print("Press 'q' to quit\n")
    
    last_data = None
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Decode Aztec codes
        results = decode(frame)
        
        # Draw results on frame
        for result in results:
            if result.type == 'AZTEC':
                # Get code position
                pts = result.polygon
                
                if len(pts) == 4:
                    # Draw code boundary
                    pts = [(point.x, point.y) for point in pts]
                    pts = np.array(pts, dtype=np.int32)
                    cv2.polylines(frame, [pts], True, (0, 255, 0), 3)
                
                # Display code data
                try:
                    data = result.data.decode('utf-8')
                except:
                    data = result.data.decode('latin-1')
                
                # Only print if it's new data
                if data != last_data:
                    print(f"Detected Aztec Code: {data}")
                    last_data = data
                
                # Display text on frame
                cv2.putText(frame, f"AZTEC: {data[:30]}", 
                           (result.rect.left, result.rect.top - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow('Aztec Code Reader (Press Q to quit)', frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def save_aztec_data(data, output_file="aztec_data.txt"):
    """
    Save decoded Aztec code data to a file
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, item in enumerate(data, 1):
                f.write(f"Aztec Code {i}:\n")
                if 'decoder' in item:
                    f.write(f"  Decoder: {item['decoder']}\n")
                f.write(f"  Data: {item['data']}\n")
                f.write("\n")
        print(f"\n✓ Data saved to {output_file}")
    except Exception as e:
        print(f"Error saving data: {e}")


# Example usage
if __name__ == "__main__":
    import sys
    
    # Replace with your image path
    image_path = "M:\\azteccode.jpg"
    
    print("="*60)
    print("AZTEC CODE READER")
    print("="*60 + "\n")
    
    # Try simple method first (pyzbar)
    print("Method 1: Using pyzbar...\n")
    data1 = read_aztec_simple(image_path)
    
    # Try ZXing (better Aztec support)
    print("\n" + "="*60)
    print("Method 2: Using ZXing-C++...\n")
    data2 = read_aztec_with_zxing(image_path)
    
    # If both fail, try advanced preprocessing
    if not data1 and not data2:
        print("\n" + "="*60)
        print("Method 3: Advanced preprocessing...\n")
        data3 = read_aztec_advanced(image_path)
    else:
        data3 = []
    
    # Combine all results
    all_data = data1 + data2 + data3
    
    # Save results if found
    if all_data:
        save_aztec_data(all_data)
        
        # Print final decoded data
        print("\n" + "="*60)
        print("FINAL DECODED DATA")
        print("="*60)
        for item in all_data:
            print(f"\n{item['data']}")
    else:
        print("\n" + "="*60)
        print("No Aztec codes could be decoded.")
        print("="*60)
        print("\nTroubleshooting tips:")
        print("1. Ensure the image is clear and well-lit")
        print("2. Try capturing at different angles")
        print("3. Make sure the entire Aztec code is visible")
        print("4. Install zxing-cpp: pip install zxing-cpp")
    
    # Uncomment to use webcam reader
    # print("\n\nStarting webcam mode...")
    # read_aztec_from_webcam()
