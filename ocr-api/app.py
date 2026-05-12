from flask import Flask, request, jsonify
from flask_cors import CORS
import easyocr
import base64
import io
import re
import numpy as np
import cv2
from PIL import Image, ExifTags

app = Flask(__name__)
CORS(app)

# Initialize EasyOCR once (slow first load, fast after)
reader = easyocr.Reader(['en'], gpu=False)

# ── Image enhancement (from try_ocr.ipynb + ocr-1.ipynb) ──
def correct_orientation(img):
    """Fix EXIF orientation like try_ocr.ipynb"""
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = img._getexif()
        if exif:
            val = exif.get(orientation)
            if val == 3:   img = img.rotate(180, expand=True)
            elif val == 6: img = img.rotate(270, expand=True)
            elif val == 8: img = img.rotate(90, expand=True)
    except:
        pass
    return img

def enhance_for_ocr(img_pil):
    """
    Full enhancement pipeline from notebooks:
    1. EXIF orientation fix
    2. Scale up 2x
    3. Grayscale
    4. CLAHE contrast enhancement
    5. Gaussian blur + Otsu threshold
    6. Sharpening
    """
    img_pil = correct_orientation(img_pil)

    # Scale up 2x for better OCR (like scale=2.0 in ocr-1.ipynb)
    w, h = img_pil.size
    img_pil = img_pil.resize((w * 2, h * 2), Image.LANCZOS)

    # Convert to OpenCV
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # Grayscale
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # CLAHE contrast enhancement (better than simple threshold)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Denoise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Sharpening kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    gray = cv2.filter2D(gray, -1, kernel)

    return gray

# ── OCR endpoint ──
@app.route('/ocr', methods=['POST'])
def ocr():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',', 1)[1]

        img_bytes = base64.b64decode(image_data)
        img_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # Enhance image
        enhanced = enhance_for_ocr(img_pil)

        # EasyOCR — returns list of (bbox, text, confidence)
        results = reader.readtext(enhanced, detail=1, paragraph=False)

        # Filter low confidence results
        lines = [r[1] for r in results if r[2] > 0.3]
        raw_text = '\n'.join(lines)

        return jsonify({
            'text': raw_text,
            'raw': raw_text,
            'lines': lines,
            'success': True
        })

    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

# ── Health check ──
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'engine': 'EasyOCR'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=False)
