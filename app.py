

import os
import io
import re
import gc
import json
import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
import pytesseract
import easyocr

# ── SVG Support ───────────────────────────────────────────────────────────────
SVG_AVAILABLE = False
try:
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPDF
    SVG_AVAILABLE = True
except:
    pass

# ── Tesseract Path (Windows) ──────────────────────────────────────────────────
if os.name == 'nt':
    for path in [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
    ]:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            break

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# ── Font setup ─────────────────────────────────────────────────────────────────────────────
DOC_FONT = 'Times-Roman'
DOC_FONT_BOLD = 'Times-Bold'

# ── OCR Text Post-Processing ──────────────────────────────────────────────────
_VALID_MINUTES = set(f'{m:02d}' for m in range(60))

def fix_ocr_text(text):
    """
    Fix OCR misread of time colon — only corrects HH.MM where:
    - hour is 0-23
    - minutes are valid (00-59)
    e.g. '2.00' -> '2:00', '10.30' -> '10:30'
    Leaves decimals like '3.14' or '99.50' untouched.
    """
    def replace_time(m):
        h, mm = m.group(1), m.group(2)
        if int(h) <= 23 and mm in _VALID_MINUTES:
            return f'{h}:{mm}'
        return m.group(0)
    return re.sub(r'\b(\d{1,2})\.(\d{2})\b', replace_time, text)

@app.after_request
def add_cors_headers(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,PUT,POST,DELETE,OPTIONS")
    return response

@app.errorhandler(Exception)
def handle_exception(e):
    import traceback
    print(f"[CRITICAL ERROR] {e}\n{traceback.format_exc()}")
    return jsonify({"error": str(e), "success": False}), 500

# ── EasyOCR: English + Filipino ───────────────────────────────────────────────
print("[OCR-API] Loading EasyOCR (en + tl)...")
try:
    easyocr_reader = easyocr.Reader(['en', 'tl'], gpu=False)
    print("[OCR-API] EasyOCR (en+tl) ready.")
except Exception as e:
    print(f"[OCR-API] Filipino model failed ({e}), falling back to English only.")
    easyocr_reader = easyocr.Reader(['en'], gpu=False)
    print("[OCR-API] EasyOCR (en) ready.")

# ── Logo Registry ─────────────────────────────────────────────────────────────
LOGOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logos')
os.makedirs(LOGOS_DIR, exist_ok=True)

LOGO_KEYWORDS = {
    "laguna university": "laguna_university",
    "lu ccs":            "laguna_university",
    "college of computer studies": "ccs_logo",
    "ccs":               "ccs_logo",
    "bachelor of science in computer science": "cs_logo",
    "bscs":              "cs_logo",
    "bachelor of science in information technology": "it_logo",
    "bsit":              "it_logo",
    "bagong pilipinas":  "bagong_pilipinas",
}

# ── Logo file mapping — exact filenames in logos/ folder ─────────────────────
LOGO_FILE_MAP = {
    'laguna_university': 'laguna_university',
    'ccs_logo':          'ccs_logo',
    'cs_logo':           'cs_logo',
    'it_logo':           'it_logo',
    'bagong_pilipinas':  'bagong_pilipinas',
}

# Keywords to detect which logos appear in the document header
LOGO_DETECT_KEYWORDS = [
    ("laguna university",                          'laguna_university'),
    ("lu-ccs",                                     'laguna_university'),
    ("lu ccs",                                     'laguna_university'),
    ("college of computer studies",                'ccs_logo'),
    ("bachelor of science in information technology", 'it_logo'),
    ("bsit",                                       'it_logo'),
    ("bachelor of science in computer science",    'cs_logo'),
    ("bscs",                                       'cs_logo'),
    ("bagong pilipinas",                           'bagong_pilipinas'),
]

def get_logo_path(key):
    """Return the actual file path for a logo key, trying common extensions."""
    filename = LOGO_FILE_MAP.get(key, key)
    for ext in ['.png', '.jpg', '.jpeg', '.webp']:
        path = os.path.join(LOGOS_DIR, filename + ext)
        if os.path.exists(path):
            return path
    return None

def detect_logos_with_position(img_bgr, ocr_results, search_ratio=0.35):
    """
    Detect logo blobs in the header region.
    Only picks blobs that are SQUARE-ISH (aspect 0.5-2.0) and NOT text-like
    (text blobs are very wide relative to height).
    Excludes any blob whose bounding box overlaps with an OCR text region
    to avoid cropping text that sits beside the logo.
    """
    img_h, img_w = img_bgr.shape[:2]
    header_limit = int(img_h * search_ratio)
    header = img_bgr[0:header_limit, :]

    gray = cv2.cvtColor(header, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Build set of OCR text bounding boxes in header region
    ocr_boxes = []
    for bbox, txt, conf in ocr_results:
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        ox1, oy1, ox2, oy2 = min(xs), min(ys), max(xs), max(ys)
        if oy1 < header_limit:
            ocr_boxes.append((ox1, oy1, ox2, oy2))

    def overlaps_text(bx, by, bw, bh):
        """Return True if this blob significantly overlaps any OCR text box."""
        for ox1, oy1, ox2, oy2 in ocr_boxes:
            ix1 = max(bx, ox1)
            iy1 = max(by, oy1)
            ix2 = min(bx + bw, ox2)
            iy2 = min(by + bh, oy2)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                blob_area = bw * bh
                # If >30% of the blob overlaps text, it IS text — skip it
                if inter / blob_area > 0.30:
                    return True
        return False

    blobs = []
    for cnt in cnts:
        bx, by, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        aspect = bw / bh if bh > 0 else 1
        # Logo: roughly square (0.5-2.0 aspect), not tiny, not full-width
        # Text lines have aspect >> 2.0, so this filters them out
        if (area > img_w * img_h * 0.0008 and
                0.5 <= aspect <= 2.0 and
                bw < img_w * 0.35 and
                bh > img_h * 0.02 and
                not overlaps_text(bx, by, bw, bh)):
            blobs.append((bx, by, bw, bh))

    blobs.sort(key=lambda b: b[0])

    found = []
    pad = 4
    for bx, by, bw, bh in blobs:
        cx = max(0, bx - pad)
        cy = max(0, by - pad)
        cw = min(img_w - cx, bw + pad * 2)
        ch = min(header_limit - cy, bh + pad * 2)
        crop = img_bgr[cy:cy+ch, cx:cx+cw]
        if crop.size == 0:
            continue

        crop_rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        ink_mask = cv2.adaptiveThreshold(
            crop_gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 8
        )
        ink_mask = cv2.dilate(ink_mask, np.ones((2, 2), np.uint8), iterations=1)
        crop_rgba[:, :, 3] = ink_mask

        success, png_buf = cv2.imencode('.png', crop_rgba)
        if not success:
            continue

        found.append({'png_bytes': png_buf.tobytes(), 'x': cx, 'y': cy, 'w': cw, 'h': ch})
        print(f"[LOGO] Cropped logo blob at ({cx},{cy}) {cw}x{ch}")

    return found


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
def get_skew_angle(cv_img):
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=5)
    contours, _ = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    angles = []
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(c) < 500: continue
        rect = cv2.minAreaRect(c)
        angle = rect[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        angles.append(angle)
    return float(np.median(angles[:10])) if angles else 0.0

def deskew_image(img):
    try:
        angle = get_skew_angle(img)
        if abs(angle) < 0.3 or abs(angle) > 25: return img
        (h, w) = img.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        cos_a, sin_a = abs(M[0,0]), abs(M[0,1])
        nw = int(h * sin_a + w * cos_a)
        nh = int(h * cos_a + w * sin_a)
        M[0,2] += (nw - w) / 2
        M[1,2] += (nh - h) / 2
        return cv2.warpAffine(img, M, (nw, nh),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(255,255,255))
    except: return img

def preprocess_for_ocr(img):
    """
    Returns (img_upscaled, sharpened_gray, bw_binary).
    img_upscaled is the COLOR image used for coordinate mapping.
    """
    img = deskew_image(img)
    h, w = img.shape[:2]
    scale = 1.0
    if w < 1600:
        scale = 2.0 if w < 800 else 1.5
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Shadow removal via background subtraction
    dilated = cv2.dilate(gray, np.ones((7,7), np.uint8))
    bg = cv2.medianBlur(dilated, 21)
    diff = 255 - cv2.absdiff(gray, bg)
    gray = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    gray = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)

    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 51, 11)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2,2), np.uint8), iterations=1)

    blurred = cv2.GaussianBlur(gray, (3,3), 0)
    sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    return img, sharpened, bw


# ═══════════════════════════════════════════════════════════════════════════════
# OCR HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def run_easyocr(image_np):
    try:
        return easyocr_reader.readtext(
            image_np, detail=1, paragraph=False,
            width_ths=0.7, height_ths=0.7,
            text_threshold=0.4, low_text=0.3,
            link_threshold=0.3
        )
    except Exception as e:
        print(f"[EasyOCR Error] {e}")
        return []

def run_tesseract_fallback(bw_image):
    try:
        return pytesseract.image_to_string(bw_image, config='--oem 3 --psm 6').strip()
    except: return ""

def assemble_text(easyocr_results):
    if not easyocr_results: return ""
    sorted_res = sorted(easyocr_results, key=lambda r: (r[0][0][1], r[0][0][0]))
    avg_h = sum(abs(r[0][0][1]-r[0][2][1]) for r in sorted_res) / len(sorted_res)
    line_thresh = max(10, avg_h * 0.6)
    lines, curr, last_y = [], [], -999
    for bbox, txt, conf in sorted_res:
        yc = (bbox[0][1] + bbox[2][1]) / 2
        if curr and abs(yc - last_y) > line_thresh:
            curr.sort(key=lambda x: x[0][0][0])
            lines.append(" ".join([t for _,t,_ in curr]))
            curr = []
        curr.append((bbox, txt, conf))
        last_y = yc
    if curr:
        curr.sort(key=lambda x: x[0][0][0])
        lines.append(" ".join([t for _,t,_ in curr]))
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# COORDINATE MAPPER — single source of truth for img→PDF coordinate conversion
# ═══════════════════════════════════════════════════════════════════════════════
class CoordMapper:
    """
    Maps pixel coordinates from the processed image to PDF points.
    PDF origin is bottom-left; image origin is top-left.

    All img_y values passed in are in PROCESSED IMAGE pixel space
    (i.e. after upscaling/deskew). The mapper never needs to know about
    the original raw image size.
    """
    def __init__(self, img_w, img_h, pdf_w, pdf_h):
        self.img_w = img_w
        self.img_h = img_h
        self.pdf_w = pdf_w
        self.pdf_h = pdf_h
        self.scale_x = pdf_w / img_w
        self.scale_y = pdf_h / img_h

    def x(self, px):
        return px * self.scale_x

    def y(self, py):
        """Convert image y (top-down, pixels) to PDF y (bottom-up, points)."""
        return self.pdf_h - (py * self.scale_y)

    def w(self, pw):
        return pw * self.scale_x

    def h(self, ph):
        return ph * self.scale_y

    def font_size(self, word_h_px):
        """
        Convert pixel height of a word bounding box to a PDF font size.
        Text typically fills ~72% of its bounding box height.
        """
        pt_h = word_h_px * self.scale_y
        return max(6.0, min(pt_h * 0.72, 48.0))

    def img_y_from_pdf_y(self, pdf_y):
        """Inverse: convert PDF y (bottom-up) back to image y (top-down, pixels)."""
        return (self.pdf_h - pdf_y) / self.scale_y


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE DETECTION & RECONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════
def detect_tables(bw, img_w, img_h):
    inv = cv2.bitwise_not(bw)

    h_kernel_len = max(40, img_w // 20)
    v_kernel_len = max(40, img_h // 20)

    h_lines_img = cv2.morphologyEx(inv, cv2.MORPH_OPEN,
                                   cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1)),
                                   iterations=2)
    v_lines_img = cv2.morphologyEx(inv, cv2.MORPH_OPEN,
                                   cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len)),
                                   iterations=2)

    table_grid = cv2.add(h_lines_img, v_lines_img)
    dilate_k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    table_grid_dilated = cv2.dilate(table_grid, dilate_k, iterations=3)

    cnts, _ = cv2.findContours(table_grid_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tables = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        tx, ty, tw, th = cv2.boundingRect(cnt)

        if tw < img_w * 0.15 or th < img_h * 0.05 or area < 2000:
            continue

        roi_h = h_lines_img[ty:ty+th, tx:tx+tw]
        roi_v = v_lines_img[ty:ty+th, tx:tx+tw]

        h_proj = np.sum(roi_h, axis=1)
        v_proj = np.sum(roi_v, axis=0)

        h_threshold = img_w * 0.10 * 255
        v_threshold = img_h * 0.03 * 255

        h_line_ys = _find_line_positions(h_proj, h_threshold, min_gap=5)
        v_line_xs = _find_line_positions(v_proj, v_threshold, min_gap=5)

        if len(h_line_ys) < 2 or len(v_line_xs) < 2:
            continue

        h_line_ys_abs = [ty + y for y in h_line_ys]
        v_line_xs_abs = [tx + x for x in v_line_xs]

        rows = []
        for r in range(len(h_line_ys) - 1):
            row = []
            for col in range(len(v_line_xs) - 1):
                cell = {
                    'x': v_line_xs_abs[col],
                    'y': h_line_ys_abs[r],
                    'w': v_line_xs_abs[col+1] - v_line_xs_abs[col],
                    'h': h_line_ys_abs[r+1]   - h_line_ys_abs[r],
                }
                row.append(cell)
            if row:
                rows.append(row)

        tables.append({
            'x': tx, 'y': ty, 'w': tw, 'h': th,
            'rows': rows,
            'h_lines': h_line_ys_abs,
            'v_lines': v_line_xs_abs,
        })

    print(f"[TABLE] Detected {len(tables)} table(s)")
    return tables


def _find_line_positions(projection, threshold, min_gap=8):
    positions = []
    in_line = False
    start = 0
    for i, val in enumerate(projection):
        if val >= threshold and not in_line:
            in_line = True
            start = i
        elif val < threshold and in_line:
            in_line = False
            center = (start + i) // 2
            if not positions or (center - positions[-1]) >= min_gap:
                positions.append(center)
    return positions


def ocr_cell(img_bgr, cell, padding=4):
    x = max(0, cell['x'] + padding)
    y = max(0, cell['y'] + padding)
    w = max(1, cell['w'] - padding * 2)
    h = max(1, cell['h'] - padding * 2)
    patch = img_bgr[y:y+h, x:x+w]
    if patch.size == 0:
        return ""
    try:
        results = easyocr_reader.readtext(patch, detail=0, paragraph=True,
                                           text_threshold=0.3, low_text=0.2)
        return " ".join(results).strip()
    except:
        try:
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            return pytesseract.image_to_string(gray, config='--psm 6 --oem 3').strip()
        except:
            return ""


def draw_table_on_pdf(c, table, img_up, mapper: CoordMapper):
    """Draw reconstructed table with accurate coordinate mapping."""
    c.setStrokeColorRGB(0, 0, 0)
    c.setLineWidth(0.7)

    tx, ty, tw, th = table['x'], table['y'], table['w'], table['h']

    # Outer border
    pdf_x = mapper.x(tx)
    pdf_y_bottom = mapper.y(ty + th)
    pdf_w = mapper.w(tw)
    pdf_h = mapper.h(th)
    c.rect(pdf_x, pdf_y_bottom, pdf_w, pdf_h, fill=0, stroke=1)

    # Inner horizontal lines
    for hy in table['h_lines'][1:-1]:
        py = mapper.y(hy)
        c.line(mapper.x(tx), py, mapper.x(tx + tw), py)

    # Inner vertical lines
    for vx in table['v_lines'][1:-1]:
        px = mapper.x(vx)
        c.line(px, mapper.y(ty), px, mapper.y(ty + th))

    # OCR each cell and draw text
    c.setFillColorRGB(0, 0, 0)
    for row in table['rows']:
        for cell in row:
            cell_text = ocr_cell(img_up, cell, padding=4)
            if not cell_text:
                continue

            cell_pdf_x  = mapper.x(cell['x']) + 3
            cell_pdf_y_bot = mapper.y(cell['y'] + cell['h'])
            cell_pdf_h  = mapper.h(cell['h'])

            # Font size proportional to cell height
            fs = max(6.0, min(cell_pdf_h * 0.55, 14.0))
            c.setFont(DOC_FONT, fs)

            # Vertically center text in cell
            text_y = cell_pdf_y_bot + (cell_pdf_h / 2) - (fs * 0.35)

            # Clip text to cell width
            cell_pdf_w = mapper.w(cell['w'])
            max_chars = max(1, int(cell_pdf_w / (fs * 0.55)))
            display_text = cell_text[:max_chars] + ('…' if len(cell_text) > max_chars else '')

            c.drawString(cell_pdf_x, text_y, display_text)

    print(f"[TABLE] Drew table @ ({tx},{ty}) rows={len(table['rows'])}")


# ═══════════════════════════════════════════════════════════════════════════════
# LINE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════
def extract_lines(bw, w, h):
    """
    Returns (h_morph, v_morph, form_lines).
    h_morph  — long horizontal dividers / headers
    v_morph  — long vertical dividers
    form_lines — short underlines used for form fields
    """
    inv = cv2.bitwise_not(bw)

    # Raise minimum length to avoid picking up design/decorative elements as ghost lines
    long_h_len = max(60, w // 30)
    h_morph = cv2.morphologyEx(inv, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (long_h_len, 1)))

    long_v_len = max(60, h // 30)
    v_morph = cv2.morphologyEx(inv, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (1, long_v_len)))

    # Form-field underlines: shorter horizontals not already captured by h_morph
    short_h_len = max(40, w // 25)
    short_h_morph = cv2.morphologyEx(inv, cv2.MORPH_OPEN,
                                     cv2.getStructuringElement(cv2.MORPH_RECT, (short_h_len, 1)))

    form_lines = cv2.subtract(short_h_morph, h_morph)

    cnts, _ = cv2.findContours(form_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_form = np.zeros_like(form_lines)
    min_form_w = int(w * 0.06)  # at least 6% of page width — stricter filter vs design artifacts
    for cnt in cnts:
        fx, fy, fw, fh = cv2.boundingRect(cnt)
        if fw >= min_form_w and fh <= 3:  # height <=3px: real underlines only
            cv2.rectangle(clean_form, (fx, fy), (fx+fw, fy+fh), 255, -1)

    return h_morph, v_morph, clean_form


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNATURE DETECTION & BACKGROUND REMOVAL
# ═══════════════════════════════════════════════════════════════════════════════
def _remove_bg(crop_bgr):
    """Remove white/light background from a crop, return BGRA with transparent bg."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    bg = cv2.dilate(gray, np.ones((7, 7), np.uint8))
    bg = cv2.medianBlur(bg, 21)
    diff = 255 - cv2.absdiff(gray, bg)
    norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    _, mask = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=1)
    rgba = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask
    return rgba


def detect_and_extract_signatures(img_bgr, ocr_results, search_ratio_start=0.30):
    """
    Detect ALL handwritten signatures.
    - Crops the signature TOGETHER with any nearby text (e.g. printed name beside sig)
      so the crop looks exactly like the original.
    - The crop area is placed as an image on the PDF — no separate text is drawn
      inside it, preventing overlap.
    - Background is removed so only ink is visible on the PDF.
    Returns list of (png_bytes, (sx, sy, sw, sh)) in absolute image coords.
    """
    img_h, img_w = img_bgr.shape[:2]
    search_start = int(img_h * search_ratio_start)
    region = img_bgr[search_start:, :]
    rh, rw = region.shape[:2]

    # --- Ink isolation ---
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    bg = cv2.dilate(gray, np.ones((7, 7), np.uint8))
    bg = cv2.medianBlur(bg, 21)
    diff = 255 - cv2.absdiff(gray, bg)
    norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    _, ink = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Dilate to merge nearby strokes into blobs
    k_blob = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    dilated = cv2.dilate(ink, k_blob, iterations=3)
    cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []

    # OCR boxes in region-relative coords
    ocr_boxes_rel = []
    for bbox, txt, conf in ocr_results:
        xs = [p[0] for p in bbox]; ys = [p[1] for p in bbox]
        ox1, oy1, ox2, oy2 = min(xs), min(ys), max(xs), max(ys)
        if oy2 >= search_start:
            ocr_boxes_rel.append((ox1, oy1 - search_start, ox2, oy2 - search_start))

    def ocr_coverage(bx, by, bw, bh):
        covered = 0
        for ox1, oy1, ox2, oy2 in ocr_boxes_rel:
            ix1 = max(bx, ox1); iy1 = max(by, oy1)
            ix2 = min(bx + bw, ox2); iy2 = min(by + bh, oy2)
            if ix2 > ix1 and iy2 > iy1:
                covered += (ix2 - ix1) * (iy2 - iy1)
        return covered / (bw * bh) if bw * bh > 0 else 0

    pending = []
    for cnt in cnts:
        bx, by, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        aspect = bw / bh if bh > 0 else 1

        if area < img_w * img_h * 0.0003:
            continue
        if bw > img_w * 0.85 or bh > img_h * 0.35:
            continue
        if aspect < 0.3 or aspect > 15:
            continue
        if ocr_coverage(bx, by, bw, bh) > 0.60:
            continue

        # Expand crop to include nearby OCR text (e.g. printed name beside sig)
        expand_margin = int(bh * 0.4)
        cx1, cy1, cx2, cy2 = bx, by, bx + bw, by + bh
        for ox1, oy1, ox2, oy2 in ocr_boxes_rel:
            if ox1 <= cx2 + expand_margin and ox2 >= cx1 - expand_margin and \
               (abs(oy1 - cy1) <= expand_margin or abs(oy2 - cy2) <= expand_margin):
                cx1 = min(cx1, ox1); cy1 = min(cy1, oy1)
                cx2 = max(cx2, ox2); cy2 = max(cy2, oy2)

        pad = 4
        cx1 = max(0, cx1 - pad); cy1 = max(0, cy1 - pad)
        cx2 = min(rw, cx2 + pad); cy2 = min(rh, cy2 + pad)
        if cx2 > cx1 and cy2 > cy1:
            pending.append((cx1, cy1, cx2 - cx1, cy2 - cy1))

    # Merge overlapping crops so 2 nearby signatures don't eat each other
    def merge_rects(rects):
        merged = list(rects)
        changed = True
        while changed:
            changed = False
            out = []
            used = [False] * len(merged)
            for i, (ax, ay, aw, ah) in enumerate(merged):
                if used[i]: continue
                ax2, ay2 = ax + aw, ay + ah
                for j, (bx2, by2, bw2, bh2) in enumerate(merged):
                    if i == j or used[j]: continue
                    bx2e, by2e = bx2 + bw2, by2 + bh2
                    if ax < bx2e and ax2 > bx2 and ay < by2e and ay2 > by2:
                        ax = min(ax, bx2); ay = min(ay, by2)
                        ax2 = max(ax2, bx2e); ay2 = max(ay2, by2e)
                        aw, ah = ax2 - ax, ay2 - ay
                        used[j] = True; changed = True
                out.append((ax, ay, aw, ah))
                used[i] = True
            merged = out
        return merged

    found = []
    for (rx, ry, rw2, rh2) in merge_rects(pending):
        abs_x = rx
        abs_y = search_start + ry
        crop = img_bgr[abs_y: abs_y + rh2, abs_x: abs_x + rw2]
        if crop.size == 0:
            continue
        rgba = _remove_bg(crop)
        ok, buf = cv2.imencode('.png', rgba)
        if not ok:
            continue
        found.append((buf.tobytes(), (abs_x, abs_y, rw2, rh2)))
        print(f"[SIGNATURE] final crop ({abs_x},{abs_y}) {rw2}x{rh2}px")

    return found


def draw_signatures_on_pdf(c, signatures, mapper: CoordMapper):
    """Overlay all extracted transparent-background signatures onto the PDF."""
    for sig_png_bytes, (sx, sy, sw, sh) in signatures:
        try:
            sig_io = io.BytesIO(sig_png_bytes)
            pdf_x = mapper.x(sx)
            pdf_y = mapper.y(sy + sh)
            pdf_w = mapper.w(sw)
            pdf_h = mapper.h(sh)
            c.drawImage(ImageReader(sig_io), pdf_x, pdf_y,
                        width=pdf_w, height=pdf_h,
                        preserveAspectRatio=False, mask='auto')
            print(f"[SIGNATURE] drew PDF ({pdf_x:.0f},{pdf_y:.0f}) {pdf_w:.0f}x{pdf_h:.0f}pt")
        except Exception as e:
            print(f"[SIGNATURE] draw error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# HANDWRITING — vector stroke tracing
# ═══════════════════════════════════════════════════════════════════════════════
def draw_handwriting_as_vectors(c, img_bgr, bw, ocr_results, table_rects,
                                mapper: CoordMapper, logo_excl_img_y=None,
                                signatures=None, logo_rects=None):
    c.setFillColorRGB(0, 0, 0)
    sig_boxes = []
    if signatures:
        for _, (sx, sy, sw, sh) in signatures:
            margin = sh * 0.12
            sig_boxes.append((sx - margin, sy - margin, sx + sw + margin, sy + sh + margin))

    for bbox, txt, conf in ocr_results:
        if not txt.strip() or conf >= 0.45:
            continue
        xs = [p[0] for p in bbox]; ys = [p[1] for p in bbox]
        x1, y1 = min(xs), min(ys)
        x2, y2 = max(xs), max(ys)
        xc = (x1 + x2) / 2; yc = (y1 + y2) / 2
        if any(tx1 <= xc <= tx2 and ty1 <= yc <= ty2 for (tx1, ty1, tx2, ty2) in table_rects):
            continue
        if logo_rects and any(lx1 <= xc <= lx2 and ly1 <= yc <= ly2 for (lx1, ly1, lx2, ly2) in logo_rects):
            continue
        if any(b[0] <= xc <= b[2] and b[1] <= yc <= b[3] for b in sig_boxes):
            continue
        fs = mapper.font_size(y2 - y1)
        c.setFont(DOC_FONT, fs)
        c.drawString(mapper.x(x1), mapper.y(y2), txt.strip())
    print(f"[HANDWRITING] Rendered handwritten text as typed")


# ═══════════════════════════════════════════════════════════════════════════════
# TEXT LINE GROUPING
# ═══════════════════════════════════════════════════════════════════════════════
def group_ocr_into_lines(ocr_results, mapper: CoordMapper, table_rects,
                         logo_excl_img_y=None, logo_rects=None, sig_rects=None):
    valid_words = []
    for bbox, txt, conf in ocr_results:
        if not txt.strip() or conf < 0.15:
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x1, y1 = min(xs), min(ys)
        x2, y2 = max(xs), max(ys)
        xc = (x1 + x2) / 2
        yc = (y1 + y2) / 2

        if any(tx1 <= xc <= tx2 and ty1 <= yc <= ty2 for (tx1, ty1, tx2, ty2) in table_rects):
            continue
        if logo_rects:
            in_logo = any(lx1 <= xc <= lx2 and ly1 <= yc <= ly2
                          for (lx1, ly1, lx2, ly2) in logo_rects)
            if in_logo:
                continue
        # Skip words whose center falls inside any signature crop region
        if sig_rects and any(sx1 <= xc <= sx2 and sy1 <= yc <= sy2 for (sx1, sy1, sx2, sy2) in sig_rects):
            continue

        valid_words.append({'text': fix_ocr_text(txt.strip()), 'x': x1, 'y_top': y1,
                            'y_bot': y2, 'y_center': yc, 'x_center': xc, 'height': y2 - y1})

    if not valid_words:
        return []

    valid_words.sort(key=lambda w: (w['y_center'], w['x']))
    heights = sorted([w['height'] for w in valid_words])
    med_h = heights[len(heights) // 2] if heights else 20
    line_gap = med_h * 0.65
    lines, current_line = [], [valid_words[0]]

    for word in valid_words[1:]:
        if abs(word['y_center'] - current_line[-1]['y_center']) <= line_gap:
            current_line.append(word)
        else:
            lines.append(current_line)
            current_line = [word]
    lines.append(current_line)

    result_lines = []
    for line_words in lines:
        line_words.sort(key=lambda w: w['x'])
        avg_y_bot = sum(w['y_bot'] for w in line_words) / len(line_words)
        avg_h = sum(w['height'] for w in line_words) / len(line_words)
        result_lines.append({'words': line_words, 'baseline_img_y': avg_y_bot,
                             'font_size': mapper.font_size(avg_h)})
    return result_lines


def draw_text_lines(c, lines, mapper: CoordMapper, signatures=None):
    c.setFillColorRGB(0, 0, 0)
    sig_boxes = []
    if signatures:
        for _, (sx, sy, sw, sh) in signatures:
            sig_boxes.append((sx - sw*0.05, sy - sh*0.05, sx + sw + sw*0.05, sy + sh + sh*0.05))
    for line in lines:
        fs = line['font_size']
        pdf_baseline = mapper.y(line['baseline_img_y'])
        c.setFont(DOC_FONT, fs)
        for word in line['words']:
            xc, yc = word['x_center'], word['y_center']
            if any(b[0] <= xc <= b[2] and b[1] <= yc <= b[3] for b in sig_boxes):
                continue
            c.drawString(mapper.x(word['x']), pdf_baseline, word['text'])


# ═══════════════════════════════════════════════════════════════════════════════
# STRUCTURAL LINE DRAWING
# ═══════════════════════════════════════════════════════════════════════════════
def draw_structural_lines(c, h_morph, v_morph, form_lines,
                           mapper: CoordMapper, table_rects,
                           logo_excl_img_y=None, logo_rects=None,
                           sig_rects=None,
                           min_h_ratio=0.05, min_v_ratio=0.04):

    def in_table_interior(cx, cy):
        margin = 8
        return any(
            (tx1 + margin) <= cx <= (tx2 - margin) and
            (ty1 + margin) <= cy <= (ty2 - margin)
            for (tx1, ty1, tx2, ty2) in table_rects
        )

    def in_logo(cx, cy):
        if not logo_rects: return False
        return any(lx1 <= cx <= lx2 and ly1 <= cy <= ly2 for (lx1, ly1, lx2, ly2) in logo_rects)

    def in_sig(cx, cy):
        if not sig_rects: return False
        return any(sx1 <= cx <= sx2 and sy1 <= cy <= sy2 for (sx1, sy1, sx2, sy2) in sig_rects)
    img_w = mapper.img_w
    img_h = mapper.img_h

    c.setStrokeColorRGB(0, 0, 0)

    c.setLineWidth(0.8)
    cnts_h, _ = cv2.findContours(h_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_drawn = 0
    for cnt in cnts_h:
        lx, ly, lw, lh = cv2.boundingRect(cnt)
        if lw / img_w < min_h_ratio: continue
        cy_img = ly + lh / 2
        cx_img = lx + lw / 2
        if in_table_interior(cx_img, cy_img): continue
        if in_logo(cx_img, cy_img): continue
        if in_sig(cx_img, cy_img): continue
        pdf_y  = mapper.y(cy_img)
        pdf_x1 = mapper.x(lx)
        pdf_x2 = mapper.x(lx + lw)
        c.line(pdf_x1, pdf_y, pdf_x2, pdf_y)
        h_drawn += 1
    print(f"[LINES] Drew {h_drawn} horizontal structural line(s)")

    c.setLineWidth(0.8)
    cnts_v, _ = cv2.findContours(v_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    v_drawn = 0
    for cnt in cnts_v:
        lx, ly, lw, lh = cv2.boundingRect(cnt)
        if lh / img_h < min_v_ratio: continue
        cx_img = lx + lw / 2
        cy_img = ly + lh / 2
        if in_table_interior(cx_img, cy_img): continue
        if in_logo(cx_img, cy_img): continue
        if in_sig(cx_img, cy_img): continue
        pdf_x     = mapper.x(cx_img)
        pdf_y_top = mapper.y(ly)
        pdf_y_bot = mapper.y(ly + lh)
        c.line(pdf_x, pdf_y_top, pdf_x, pdf_y_bot)
        v_drawn += 1
    print(f"[LINES] Drew {v_drawn} vertical structural line(s)")

    c.setLineWidth(0.5)
    cnts_f, _ = cv2.findContours(form_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    f_drawn = 0
    for cnt in cnts_f:
        lx, ly, lw, lh = cv2.boundingRect(cnt)
        cx_img = lx + lw / 2
        cy_img = ly + lh / 2
        if in_table_interior(cx_img, cy_img): continue
        if in_logo(cx_img, cy_img): continue
        if in_sig(cx_img, cy_img): continue
        pdf_y  = mapper.y(cy_img)
        pdf_x1 = mapper.x(lx)
        pdf_x2 = mapper.x(lx + lw)
        c.line(pdf_x1, pdf_y, pdf_x2, pdf_y)
        f_drawn += 1
    if f_drawn:
        print(f"[LINES] Drew {f_drawn} form-field underline(s)")


















# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'online',
        'version': '8.0.0',
        'service': 'lu-ccs-ocr-api',
        'svg_support': SVG_AVAILABLE,
        'ocr_engine': 'easyocr(en+tl)+tesseract',
        'logos_available': [k for k in LOGO_FILE_MAP if get_logo_path(k)]
    }), 200


@app.route('/ocr-image', methods=['POST'])
def ocr_image():
    try:
        data = request.get_json(force=True)
        image_b64 = data.get('image', '')
        if not image_b64:
            return jsonify({'error': 'Missing image data', 'success': False}), 400
        if ',' in image_b64:
            image_b64 = image_b64.split(',', 1)[1]
        raw = cv2.imdecode(np.frombuffer(base64.b64decode(image_b64), np.uint8), cv2.IMREAD_COLOR)
        if raw is None:
            return jsonify({'error': 'Invalid image format', 'success': False}), 400
        img_up, sharpened, bw = preprocess_for_ocr(raw)
        results = run_easyocr(img_up)
        text = assemble_text(results) if results else run_tesseract_fallback(bw)
        return jsonify({'text': text, 'success': True, 'version': '6.2.0'})
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/reconstruct', methods=['POST'])
def reconstruct():
    try:
        data = request.get_json(force=True)
        images_b64 = data.get('images', [])
        print(f"[RECONSTRUCT V8.0] {len(images_b64)} page(s)")

        pdf_io = io.BytesIO()
        c = canvas.Canvas(pdf_io, pagesize=A4)
        all_pages_text = []

        for idx, b64 in enumerate(images_b64):
            b64_data = b64.split(',')[-1] if ',' in b64 else b64
            raw = cv2.imdecode(np.frombuffer(base64.b64decode(b64_data), np.uint8), cv2.IMREAD_COLOR)
            if raw is None:
                print(f"[PAGE {idx+1}] Decode failed, skipping.")
                continue

            # Preprocess
            img_up, sharpened, bw = preprocess_for_ocr(raw)
            proc_h, proc_w = img_up.shape[:2]

            PDF_W = 612.0
            PDF_H = PDF_W * (proc_h / proc_w)
            c.setPageSize((PDF_W, PDF_H))
            mapper = CoordMapper(proc_w, proc_h, PDF_W, PDF_H)

            # White background
            c.setFillColorRGB(1, 1, 1)
            c.rect(0, 0, PDF_W, PDF_H, fill=1, stroke=0)

            # OCR
            ocr_results = run_easyocr(img_up)
            if not ocr_results:
                fallback = run_tesseract_fallback(bw)
                all_pages_text.append(fallback)
                c.setFillColorRGB(0, 0, 0)
                c.setFont(DOC_FONT, 10)
                tb = c.beginText(40, PDF_H - 50)
                for line in fallback.split('\n'):
                    tb.textLine(line)
                c.drawText(tb)
                c.showPage()
                continue

            # Detect tables
            tables = detect_tables(bw, proc_w, proc_h)
            table_rects = [(t['x'], t['y'], t['x']+t['w'], t['y']+t['h']) for t in tables]

            # Extract structural lines
            h_morph, v_morph, form_lines = extract_lines(bw, proc_w, proc_h)

            # ── Logo detection — crop + background removal (any logo) ──────────
            logo_items = detect_logos_with_position(img_up, ocr_results, search_ratio=0.35)
            logo_excl_img_y = None
            logo_rects = []  # (x1, y1, x2, y2) per logo in image coords

            if logo_items:
                for item in logo_items:
                    try:
                        logo_io = io.BytesIO(item['png_bytes'])
                        pdf_x = mapper.x(item['x'])
                        pdf_y = mapper.y(item['y'] + item['h'])
                        pdf_w = mapper.w(item['w'])
                        pdf_h = mapper.h(item['h'])
                        c.drawImage(ImageReader(logo_io), pdf_x, pdf_y,
                                    width=pdf_w, height=pdf_h,
                                    preserveAspectRatio=False, mask='auto')
                        logo_rects.append((
                            item['x'], item['y'],
                            item['x'] + item['w'], item['y'] + item['h']
                        ))
                        print(f"[LOGO] Drew cropped logo at ({pdf_x:.0f},{pdf_y:.0f}) {pdf_w:.0f}x{pdf_h:.0f}pt")
                    except Exception as e:
                        print(f"[LOGO] Draw error: {e}")

            # Draw tables
            for tbl in tables:
                draw_table_on_pdf(c, tbl, img_up, mapper)

            # Detect & extract ALL signatures FIRST so lines can exclude sig areas
            signatures = detect_and_extract_signatures(img_up, ocr_results)
            sig_rects = [(sx - sw*0.05, sy - sh*0.05, sx+sw + sw*0.05, sy+sh + sh*0.05)
                         for _, (sx, sy, sw, sh) in signatures]

            # Draw structural lines (excluding signature regions to prevent ghost lines)
            draw_structural_lines(c, h_morph, v_morph, form_lines,
                                   mapper, table_rects, logo_excl_img_y=None,
                                   logo_rects=logo_rects, sig_rects=sig_rects)

            # Draw text (typed + handwritten) excluding all signature regions
            text_lines = group_ocr_into_lines(
                ocr_results, mapper, table_rects, logo_rects=logo_rects, sig_rects=sig_rects)
            draw_text_lines(c, text_lines, mapper, signatures=signatures)

            # Overlay all signatures onto PDF
            draw_signatures_on_pdf(c, signatures, mapper)

            all_pages_text.append(assemble_text(ocr_results))
            c.showPage()

            del img_up, sharpened, bw, raw
            gc.collect()

        c.save()
        return jsonify({
            'success': True,
            'pdf_base64': base64.b64encode(pdf_io.getvalue()).decode('utf-8'),
            'full_text': "\n\n".join(all_pages_text),
            'version': '8.0.0'
        })

    except Exception as e:
        import traceback
        print(f"[RECONSTRUCT ERROR] {e}\n{traceback.format_exc()}")
        return jsonify({'error': str(e), 'success': False}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5050)), debug=False)
