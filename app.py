from flask import Flask, request, jsonify
from flask_cors import CORS
import easyocr
import base64
import io
import os
import re
from PIL import Image
from fpdf import FPDF

app = Flask(__name__)

# Configure CORS to allow documnettracker.com and local dev
CORS(app, resources={r"/*": {
    "origins": [
        "https://documnettracker.com",
        "https://lu-ccs-ldts.web.app",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ]
}})

# Initialize EasyOCR (English only to save RAM on 512MB instance)
print("🚀 Initializing EasyOCR Reader...")
reader = easyocr.Reader(['en'], gpu=False)

def clean_ocr_text(text):
    # Keep characters, numbers, and basic punctuation
    text = re.sub(r'[^a-zA-Z0-9\s\.,!\?\-\(\)]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "service": "Ultimate Fresh Type Engine v6.1.0"}), 200

@app.route('/ocr-image', methods=['POST'])
def ocr_image():
    try:
        data = request.get_json()
        image_data = data.get('image', '')

        if ',' in image_data:
            image_data = image_data.split(',', 1)[1]

        img_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        results = reader.readtext(img, detail=0, paragraph=True)
        raw_text = '\n'.join(results)
        
        return jsonify({'text': raw_text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/reconstruct', methods=['POST'])
def reconstruct():
    try:
        data = request.get_json()
        images = data.get('images', [])
        
        if not images:
            return jsonify({'error': 'No images provided'}), 400

        # PDF setup (A4 size)
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)

        for img_base64 in images:
            if ',' in img_base64:
                img_base64 = img_base64.split(',', 1)[1]

            img_bytes = base64.b64decode(img_base64)
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            
            # Save temp image for FPDF
            temp_path = "temp_page.jpg"
            img.save(temp_path, "JPEG", quality=95)

            # OCR for coordinate mapping
            pdf.add_page()
            results = reader.readtext(img)

            # Draw "Fresh" digital text over a white canvas
            # (Note: For true "Fresh Type", we often skip the original image 
            # and only place the detected text and structural lines)
            
            # Set font for reconstruction
            pdf.set_font("Helvetica", size=10)
            
            # Scaling factors (A4 is 210x297mm)
            img_w, img_h = img.size
            scale_x = 210 / img_w
            scale_y = 297 / img_h

            for (bbox, text, prob) in results:
                if prob < 0.2: continue
                # bbox: [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]
                x = bbox[0][0] * scale_x
                y = bbox[0][1] * scale_y
                
                pdf.set_xy(x, y)
                pdf.cell(0, 10, text)

            if os.path.exists(temp_path):
                os.remove(temp_path)

        # Output to base64
        pdf_output = pdf.output(dest='S')
        pdf_base64 = base64.b64encode(pdf_output).decode('utf-8')

        return jsonify({'pdf_base64': pdf_base64})
    except Exception as e:
        print(f"❌ Reconstruction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    # Listen on all interfaces for Render
    app.run(host='0.0.0.0', port=port)
