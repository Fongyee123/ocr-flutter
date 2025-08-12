from flask import Flask, request, jsonify
import cv2
import numpy as np
from paddleocr import PaddleOCR
import paddle
from flask_cors import CORS
import re
import difflib
import json # Import json for pretty printing dicts

# Force PaddleOCR to use CPU
paddle.set_device('cpu')

app = Flask(__name__)
CORS(app)

# Initialize OCR
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)

# --- Helper Functions with Debugging ---

def normalize_text(text):
    """Removes punctuation, converts to lowercase, handles extra spaces."""
    if not isinstance(text, str): # Basic type check
        text = str(text)
    text_no_punct = re.sub(r'[^\w\s]', '', text)
    text_single_space = re.sub(r'\s+', ' ', text_no_punct).strip()
    lower_text = text_single_space.lower()
    # print(f"DEBUG [normalize_text] Output: '{lower_text[:100]}...'") # Keep debug optional
    return lower_text

def word_accuracy(ocr_text, ground_truth):
    """Calculates word-level similarity ratio."""
    # print("\nDEBUG [word_accuracy] Calculating...") # Keep debug optional
    ocr_words = normalize_text(ocr_text).split()
    gt_words = normalize_text(ground_truth).split()
    if not gt_words and not ocr_words: return 1.0
    if not gt_words or not ocr_words: return 0.0
    ratio = difflib.SequenceMatcher(None, ocr_words, gt_words).ratio()
    # print(f"DEBUG [word_accuracy] Ratio: {ratio:.4f}") # Keep debug optional
    return ratio

def char_accuracy(ocr_text, ground_truth):
    """Calculates character-level similarity ratio."""
    # print("\nDEBUG [char_accuracy] Calculating...") # Keep debug optional
    ocr_text_norm = normalize_text(ocr_text)
    ground_truth_norm = normalize_text(ground_truth)
    if not ground_truth_norm and not ocr_text_norm: return 1.0
    if not ground_truth_norm or not ocr_text_norm: return 0.0
    ratio = difflib.SequenceMatcher(None, ocr_text_norm, ground_truth_norm).ratio()
    # print(f"DEBUG [char_accuracy] Ratio: {ratio:.4f}") # Keep debug optional
    return ratio

def error_count(ocr_text, ground_truth):
    """Calculates word and character edit distance using SequenceMatcher."""
    # print("\nDEBUG [error_count] Calculating...") # Keep debug optional
    ocr_words = normalize_text(ocr_text).split()
    gt_words = normalize_text(ground_truth).split()

    matcher_w = difflib.SequenceMatcher(None, ocr_words, gt_words)
    tp_w = sum(block.size for block in matcher_w.get_matching_blocks())
    word_errors = len(ocr_words) + len(gt_words) - 2 * tp_w
    # print(f"DEBUG [error_count] Word TP: {tp_w}, Word Errors: {word_errors}") # Keep debug optional

    ocr_text_normalized = normalize_text(ocr_text)
    gt_text_normalized = normalize_text(ground_truth)
    matcher_c = difflib.SequenceMatcher(None, ocr_text_normalized, gt_text_normalized)
    tp_c = sum(block.size for block in matcher_c.get_matching_blocks())
    char_errors = len(ocr_text_normalized) + len(gt_text_normalized) - 2 * tp_c
    # print(f"DEBUG [error_count] Char TP: {tp_c}, Char Errors: {char_errors}") # Keep debug optional

    return word_errors, char_errors

def calculate_prf_word_level(ocr_text, ground_truth):
    """Calculates word-level Precision, Recall, F1 using SequenceMatcher."""
    # print("\nDEBUG [calculate_prf_word_level] Calculating...") # Keep debug optional
    ocr_words = normalize_text(ocr_text).split()
    gt_words = normalize_text(ground_truth).split()

    if not gt_words:
        res = {'precision': 100.0, 'recall': 100.0, 'f1_score': 100.0} if not ocr_words else {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
        # print(f"DEBUG [PRF] GT Empty. Result: {res}") # Keep debug optional
        return res
    if not ocr_words:
        res = {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
        # print(f"DEBUG [PRF] OCR Empty. Result: {res}") # Keep debug optional
        return res

    matcher = difflib.SequenceMatcher(None, ocr_words, gt_words)
    tp = sum(block.size for block in matcher.get_matching_blocks())
    fp = len(ocr_words) - tp
    fn = len(gt_words) - tp
    # print(f"DEBUG [PRF] TP: {tp}, FP: {fp}, FN: {fn}") # Keep debug optional

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    result = {
        'precision': round(precision * 100, 2),
        'recall': round(recall * 100, 2),
        'f1_score': round(f1 * 100, 2)
    }
    # print(f"DEBUG [PRF] Result: {result}") # Keep debug optional
    return result

def calculate_overall_metrics(ocr_text, ground_truth):
    """Calculates all metrics."""
    # print("\nDEBUG [calculate_overall_metrics] Calculating all metrics...") # Keep debug optional
    # print(f"DEBUG [calculate_overall_metrics] OCR Input: '{ocr_text[:80]}...'") # Keep debug optional
    # print(f"DEBUG [calculate_overall_metrics] GT Input: '{ground_truth[:80]}...'") # Keep debug optional

    metrics = {
        'word_accuracy_percent': round(word_accuracy(ocr_text, ground_truth) * 100, 2),
        'char_accuracy_percent': round(char_accuracy(ocr_text, ground_truth) * 100, 2),
        # REMOVED Similarity Ratio - It was redundant with char_accuracy_percent
        # 'similarity_ratio_percent': round(char_accuracy(ocr_text, ground_truth) * 100, 2)
    }

    word_errors, char_errors = error_count(ocr_text, ground_truth)
    metrics['word_errors'] = word_errors
    metrics['char_errors'] = char_errors

    prf_metrics = calculate_prf_word_level(ocr_text, ground_truth)
    metrics.update(prf_metrics)

    # print(f"DEBUG [calculate_overall_metrics] Final Metrics: {json.dumps(metrics, indent=2)}") # Keep debug optional
    return metrics

def extract_text_and_metrics(image, image_desc="image", ground_truth=None): # Added image_desc for debug
    """Extracts text using OCR and calculates metrics if ground truth is provided."""
    # print(f"\nDEBUG [extract_text_and_metrics] Processing {image_desc}...") # Keep debug optional
    ocr_text, words_info, metrics = "", [], {} # Initialize defaults
    try:
        result = ocr.ocr(image, cls=True)
        all_text = []
        words_info = []

        if result is not None and result[0] is not None:
            # print(f"DEBUG [extract_text_and_metrics] Raw OCR result structure: {type(result)}, {type(result[0])}, len: {len(result[0]) if isinstance(result[0], list) else 'N/A'}") # Keep debug optional
             for line_result in result[0]:
                if line_result is None: continue
                if len(line_result) == 2 and isinstance(line_result[0], list) and isinstance(line_result[1], tuple) and len(line_result[0]) == 4 and len(line_result[1]) == 2:
                    box = line_result[0]
                    text_info = line_result[1]
                    if all(isinstance(pt, list) and len(pt) == 2 for pt in box):
                        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = box
                    else: continue # Skip malformed box

                    text = text_info[0]
                    conf = text_info[1]
                    try:
                        conf_float = round(float(conf) * 100, 2)
                    except (ValueError, TypeError): continue # Skip non-numeric confidence

                    if conf_float >= 50: # Confidence threshold
                        all_text.append(text)
                        words_info.append({ #... word info structure ...
                            'text': text, 'confidence': conf_float,
                            'bounding_box': {'top_left': [int(x1), int(y1)], 'top_right': [int(x2), int(y2)], 'bottom_right': [int(x3), int(y3)], 'bottom_left': [int(x4), int(y4)]}
                        })
                # else: print(f"DEBUG [extract_text_and_metrics] Skipping unexpected line format: {line_result}") # Keep debug optional
             ocr_text = ' '.join(all_text)
             # print(f"DEBUG [extract_text_and_metrics] Extracted text ({image_desc}): '{ocr_text[:100]}...'") # Keep debug optional
        else:
             # print(f"DEBUG [extract_text_and_metrics] OCR returned None or empty result for {image_desc}.") # Keep debug optional
             ocr_text = ""

    except Exception as e:
        print(f"ERROR [extract_text_and_metrics] Error during OCR for {image_desc}: {e}")
        ocr_text = "" # Ensure it's empty on error
        words_info = []

    if ground_truth and ground_truth.strip():
        # print(f"DEBUG [extract_text_and_metrics] Calculating metrics for {image_desc} against GT...") # Keep debug optional
        metrics = calculate_overall_metrics(ocr_text, ground_truth)
    else:
        # print(f"DEBUG [extract_text_and_metrics] No ground truth provided for {image_desc}, skipping metrics calculation.") # Keep debug optional
        metrics = { # Default empty metrics structure (similarity_ratio removed)
            'word_accuracy_percent': None,
            'char_accuracy_percent': None,
            # 'similarity_ratio_percent': None, # REMOVED
            'word_errors': None,
            'char_errors': None,
            'precision': None,
            'recall': None,
            'f1_score': None
        }

    return ocr_text, words_info, metrics

def preprocess_image(image_cv):
    """Applies preprocessing steps."""
    # print("\nDEBUG [preprocess_image] Starting...") # Keep debug optional
    if image_cv is None: return None
    try:
        # Grayscale
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        # print("DEBUG [preprocess_image] Converted to grayscale.") # Keep debug optional
        # Gaussian Blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # print("DEBUG [preprocess_image] Applied Gaussian Blur (5x5).") # Keep debug optional
        # Resize (Original Color)
        if image_cv.shape[0] > 0 and image_cv.shape[1] > 0:
             image_resized = cv2.resize(image_cv, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
             # print(f"DEBUG [preprocess_image] Resized original color image by 1.2x to {image_resized.shape[:2]}.") # Keep debug optional
        else: image_resized = image_cv
        # Morphological Closing (Resized Color)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(image_resized, cv2.MORPH_CLOSE, kernel)
        # print("DEBUG [preprocess_image] Applied Morphological Closing (3x3 kernel) on resized color image.") # Keep debug optional
        # print("DEBUG [preprocess_image] Finished.") # Keep debug optional
        return cleaned
    except Exception as e:
        print(f"ERROR [preprocess_image] Error: {e}")
        return image_cv # Return original on error

# --- Flask Routes ---

@app.route('/paddle-ocr', methods=['POST'])
def upload_image():
    # print("\n--- Request Received: /paddle-ocr ---") # Keep debug optional
    if 'image' not in request.files:
        print("ERROR: No image file in request.")
        return jsonify({'error': 'No image file found in the request'}), 400
    file = request.files['image']
    # print(f"DEBUG: Received file: {file.filename}") # Keep debug optional

    try:
        image_bytes = np.frombuffer(file.read(), np.uint8)
        original_image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        if original_image is None: # Try alpha channel
            original_image = cv2.imdecode(image_bytes, cv2.IMREAD_UNCHANGED)
            if original_image is not None and len(original_image.shape) > 2 and original_image.shape[2] == 4:
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGRA2BGR)
            elif original_image is None:
                 print("ERROR: Invalid image format after attempting decode.")
                 return jsonify({'error': 'Invalid or unsupported image format'}), 400
        # print(f"DEBUG: Image decoded successfully. Shape: {original_image.shape}") # Keep debug optional
    except Exception as e:
        print(f"ERROR: Failed to read/decode image: {e}")
        return jsonify({'error': f'Failed to read or decode image: {str(e)}'}), 400

    test_plan = request.form.get('test_plan', '').lower()
    ground_truth = request.form.get('ground_truth', '').strip()
    # print(f"DEBUG: Test Plan: '{test_plan}', Ground Truth Provided: {bool(ground_truth)}") # Keep debug optional
    # if ground_truth: print(f"DEBUG: Ground Truth: '{ground_truth[:100]}...'") # Keep debug optional

    response = {'test_plan': test_plan, 'ground_truth': ground_truth}

    # --- Before Preprocessing ---
    ocr_text_before, words_before, metrics_before = "", [], {}
    if ground_truth:
        # print("\nDEBUG: == Running OCR on Original Image ==") # Keep debug optional
        ocr_text_before, words_before, metrics_before = extract_text_and_metrics(original_image, "Original", ground_truth)
    # else: print("\nDEBUG: Skipping OCR on Original Image (no ground truth).") # Keep debug optional


    # --- Preprocessing ---
    # print("\nDEBUG: == Preprocessing Image ==") # Keep debug optional
    cleaned_image = preprocess_image(original_image)
    if cleaned_image is None:
        print("ERROR: Preprocessing returned None.")
        return jsonify({'error': 'Image preprocessing failed'}), 500

    # --- After Preprocessing ---
    # print("\nDEBUG: == Running OCR on Preprocessed Image ==") # Keep debug optional
    ocr_text_after, words_after, metrics_after = extract_text_and_metrics(cleaned_image, "Preprocessed", ground_truth if ground_truth else None)

    # --- Structuring Response ---
    if test_plan == 'test3' and ground_truth:
        # print("\nDEBUG: Structuring response for Test Plan 3 (with GT)...") # Keep debug optional
        wa_before = metrics_before.get('word_accuracy_percent')
        wa_after = metrics_after.get('word_accuracy_percent')
        improvement = None
        if wa_before is not None and wa_after is not None:
             improvement = round(wa_after - wa_before, 2)
             # print(f"DEBUG: Accuracy Improvement: {improvement:.2f}% (After: {wa_after}%, Before: {wa_before}%)") # Keep debug optional
        # else: print("DEBUG: Cannot calculate improvement (missing before/after accuracy).") # Keep debug optional

        response.update({
            'before_preprocessing': {'ocr_text': ocr_text_before, 'words': words_before, 'metrics': metrics_before},
            'after_preprocessing': {'ocr_text': ocr_text_after, 'words': words_after, 'metrics': metrics_after},
            'accuracy_improvement': improvement,
            'message': 'Evaluation completed successfully.'
        })
    else:
        # print("\nDEBUG: Structuring response for other Test Plans or no GT...") # Keep debug optional
        response.update({
            'ocr_text': ocr_text_after, 'words': words_after, 'metrics': metrics_after,
            'message': 'OCR processing complete.' + (' Please provide ground truth for evaluation.' if not ground_truth else '')
        })
        response.setdefault('before_preprocessing', {'ocr_text': '', 'words': [], 'metrics': {}})
        response.setdefault('accuracy_improvement', None)

    # print(f"\nDEBUG: == Sending Response ==\n{json.dumps(response, indent=2)}\n-----------------------------") # Keep debug optional
    return jsonify(response)


@app.route('/evaluate-ocr', methods=['POST'])
def evaluate_with_ground_truth():
    # print("\n--- Request Received: /evaluate-ocr ---") # Keep debug optional
    ground_truth = request.form.get('ground_truth', '').strip()
    ocr_text = request.form.get('ocr_text', '')
    # print(f"DEBUG: Received GT: '{ground_truth[:80]}...'") # Keep debug optional
    # print(f"DEBUG: Received OCR Text: '{ocr_text[:80]}...'") # Keep debug optional

    if not ground_truth or not ocr_text:
        print("ERROR: Missing GT or OCR text.")
        return jsonify({'error': 'Missing OCR text or ground truth for evaluation'}), 400

    metrics = calculate_overall_metrics(ocr_text, ground_truth)

    response = {'ground_truth': ground_truth, 'ocr_text': ocr_text, 'metrics': metrics}
    # print(f"\nDEBUG: == Sending Response ==\n{json.dumps(response, indent=2)}\n-----------------------------") # Keep debug optional
    return jsonify(response)


@app.route('/paddle-ocr', methods=['POST'])
def ocr_words_only():
    # print("\n--- Request Received: /paddle-ocr-words ---") # Keep debug optional
    if 'image' not in request.files: return jsonify({'error': 'No image file found'}), 400
    file = request.files['image']; # print(f"DEBUG: Received file: {file.filename}") # Keep debug optional
    try:
        image_bytes = np.frombuffer(file.read(), np.uint8); original_image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        if original_image is None: return jsonify({'error': 'Invalid image format'}), 400
        # print(f"DEBUG: Image decoded. Shape: {original_image.shape}") # Keep debug optional
    except Exception as e: print(f"ERROR: Image decode failed: {e}"); return jsonify({'error': f'Image decode failed: {e}'}),400

    # print("\nDEBUG: == Preprocessing Image ==") # Keep debug optional
    cleaned_image = preprocess_image(original_image)
    if cleaned_image is None: return jsonify({'error': 'Preprocessing failed'}), 500

    # print("\nDEBUG: == Running OCR on Preprocessed Image (Words Only) ==") # Keep debug optional
    ocr_text, words_info, _ = extract_text_and_metrics(cleaned_image, "Preprocessed (Words Only)", ground_truth=None)

    response = {'ocr_text': ocr_text, 'words': words_info, 'message': 'OCR words extracted successfully (no evaluation).'}
    # print(f"\nDEBUG: == Sending Response ==\n{json.dumps(response, indent=2)}\n-----------------------------") # Keep debug optional
    return jsonify(response)


if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(host="0.0.0.0", port=5000, debug=True)