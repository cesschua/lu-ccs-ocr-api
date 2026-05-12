from flask import Flask, request, jsonify
from flask_cors import CORS
import pytesseract
import base64
import io
import numpy as np
import cv2
from PIL import Image, ExifTags

app = Flask(__name__)
CORS(app)

def correct_orientation(img):
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
    img_pil = correct_orientation(img_pil)
    w, h = img_pil.size
    img_pil = img_pil.resize((w * 2, h * 2), Image.LANCZOS)
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    gray = cv2.filter2D(gray, -1, kernel)
    return gray

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
        enhanced = enhance_for_ocr(img_pil)
        enhanced_pil = Image.fromarray(enhanced)

        raw_text = pytesseract.image_to_string(enhanced_pil, config='--psm 6')
        lines = [l.strip() for l in raw_text.splitlines() if l.strip()]

        return jsonify({
            'text': raw_text,
            'raw': raw_text,
            'lines': lines,
            'success': True
        })

    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'engine': 'Tesseract'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=False)
