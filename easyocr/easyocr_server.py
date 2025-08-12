from flask import Flask, request, jsonify
import cv2
import numpy as np
import easyocr  # Replace PaddleOCR with EasyOCR
from flask_cors import CORS
import re
import difflib
import json

app = Flask(__name__)
CORS(app)

# Initialize EasyOCR (English only)
reader = easyocr.Reader(['en'])

def normalize_text(text):
    """Removes punctuation, converts to lowercase, handles extra spaces."""
    if not isinstance(text, str):
        text = str(text)
    text_no_punct = re.sub(r'[^\w\s]', '', text)
    text_single_space = re.sub(r'\s+', ' ', text_no_punct).strip()
    lower_text = text_single_space.lower()
    return lower_text

# --- Accuracy/Ratio Functions ---
def word_accuracy(ocr_text, ground_truth):
    ocr_words = normalize_text(ocr_text).split()
    gt_words = normalize_text(ground_truth).split()
    if not gt_words and not ocr_words: return 1.0
    if not gt_words or not ocr_words: return 0.0
    ratio = difflib.SequenceMatcher(None, ocr_words, gt_words).ratio()
    return ratio

def char_accuracy(ocr_text, ground_truth):
    ocr_text_norm = normalize_text(ocr_text)
    ground_truth_norm = normalize_text(ground_truth)
    if not ground_truth_norm and not ocr_text_norm: return 1.0
    if not ground_truth_norm or not ocr_text_norm: return 0.0
    ratio = difflib.SequenceMatcher(None, ocr_text_norm, ground_truth_norm).ratio()
    return ratio


# --- Calculate ALL metrics ---
def calculate_overall_metrics(ocr_text, ground_truth):
    ocr_text_norm = normalize_text(ocr_text)
    gt_text_norm = normalize_text(ground_truth)
    ocr_words_norm = ocr_text_norm.split()
    gt_words_norm = gt_text_norm.split()

    # Basic Counts
    total_ocr_words = len(ocr_words_norm)
    total_ocr_chars = len(ocr_text_norm)
    total_gt_words = len(gt_words_norm)
    total_gt_chars = len(gt_text_norm)

    metrics = {
        'total_ocr_words': total_ocr_words,
        'total_ocr_chars': total_ocr_chars,
        'total_gt_words': total_gt_words,
        'total_gt_chars': total_gt_chars,
    }

    # Matching and Error Calculation
    matched_words = 0
    matched_chars = 0
    word_errors = 0
    char_errors = 0
    ocr_unmatched_words = 0
    ocr_unmatched_chars = 0
    gt_unmatched_words = 0
    gt_unmatched_chars = 0

    # Word matches/errors
    if total_gt_words > 0 or total_ocr_words > 0:
        matcher_w = difflib.SequenceMatcher(None, ocr_words_norm, gt_words_norm)
        matched_words = sum(block.size for block in matcher_w.get_matching_blocks())
        word_errors = total_ocr_words + total_gt_words - 2 * matched_words
        ocr_unmatched_words = total_ocr_words - matched_words
        gt_unmatched_words = total_gt_words - matched_words

    # Character matches/errors
    if total_gt_chars > 0 or total_ocr_chars > 0:
        matcher_c = difflib.SequenceMatcher(None, ocr_text_norm, gt_text_norm)
        matched_chars = sum(block.size for block in matcher_c.get_matching_blocks())
        char_errors = total_ocr_chars + total_gt_chars - 2 * matched_chars
        ocr_unmatched_chars = total_ocr_chars - matched_chars
        gt_unmatched_chars = total_gt_chars - matched_chars

    metrics.update({
        'matched_words': matched_words,
        'matched_chars': matched_chars,
        'gt_unmatched_words': gt_unmatched_words,
        'gt_unmatched_chars': gt_unmatched_chars,
        'ocr_unmatched_words': ocr_unmatched_words,
        'ocr_unmatched_chars': ocr_unmatched_chars,
        'word_errors': word_errors,
        'char_errors': char_errors,
    })

    # Accuracy Ratios
    metrics.update({
        'word_accuracy_ratio_percent': round(word_accuracy(ocr_text, ground_truth) * 100, 2),
        'char_accuracy_ratio_percent': round(char_accuracy(ocr_text, ground_truth) * 100, 2),
    })

    # Precision, Recall, F1 (Word Level)
    tp_w = matched_words
    fp_w = ocr_unmatched_words
    fn_w = gt_unmatched_words   

    # Precision, Recall, F1 (Character Level)
    tp_c = matched_chars
    fp_c = ocr_unmatched_chars
    fn_c = gt_unmatched_chars

    precision_w = tp_w / (tp_w + fp_w) if (tp_w + fp_w) > 0 else (1.0 if total_gt_words == 0 and total_ocr_words == 0 else 0.0)
    recall_w = tp_w / (tp_w + fn_w) if (tp_w + fn_w) > 0 else (1.0 if total_gt_words == 0 and total_ocr_words == 0 else 0.0)
    f1_w = 2 * (precision_w * recall_w) / (precision_w + recall_w) if (precision_w + recall_w) > 0 else (1.0 if total_gt_words == 0 and total_ocr_words == 0 else 0.0)

    precision_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else (1.0 if total_gt_chars == 0 and total_ocr_chars == 0 else 0.0)
    recall_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else (1.0 if total_gt_chars == 0 and total_ocr_chars == 0 else 0.0)
    f1_c = 2 * (precision_c * recall_c) / (precision_c + recall_c) if (precision_c + recall_c) > 0 else (1.0 if total_gt_chars == 0 and total_ocr_chars == 0 else 0.0)


    metrics.update({
        'precision': round(precision_w * 100, 2),
        'recall': round(recall_w * 100, 2),
        'f1_score': round(f1_w * 100, 2),
        'precision_c': round(precision_c * 100, 2),
        'recall_c': round(recall_c * 100, 2),
        'f1_score_c': round(f1_c * 100, 2)
    })

    return metrics

# --- Default Metrics Structure ---
DEFAULT_METRICS = {
    'total_ocr_words': None, 'total_ocr_chars': None,
    'total_gt_words': None, 'total_gt_chars': None,
    'matched_words': None, 'matched_chars': None,
    'gt_unmatched_words': None, 'gt_unmatched_chars': None,
    'ocr_unmatched_words': None, 'ocr_unmatched_chars': None,
    'word_errors': None, 'char_errors': None,
    'word_accuracy_ratio_percent': None, 'char_accuracy_ratio_percent': None,
    'precision': None, 'recall': None, 'f1_score': None,
    'precision_c': None, 'recall_c': None, 'f1_score_c': None
}

# --- Extract Text and Metrics (EasyOCR Version) ---
def extract_text_and_metrics(image, image_desc="image", ground_truth=None):
    ocr_text, words_info, metrics = "", [], {}
    try:
        # EasyOCR returns a list of (bbox, text, confidence) tuples
        result = reader.readtext(image)
        all_text = []
        words_info = []

        for (bbox, text, confidence) in result:
            if confidence >= 0.5:  # Confidence threshold (0-1)
                all_text.append(text)
                # Convert bbox to [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                bbox_list = [[int(pt[0]), int(pt[1])] for pt in bbox]
                words_info.append({
                    'text': text,
                    'confidence': round(confidence * 100, 2),
                    'bounding_box': {
                        'top_left': bbox_list[0],
                        'top_right': bbox_list[1],
                        'bottom_right': bbox_list[2],
                        'bottom_left': bbox_list[3]
                    }
                })

        ocr_text = ' '.join(all_text)
    except Exception as e:
        print(f"ERROR [extract_text_and_metrics] Error during OCR for {image_desc}: {e}")
        ocr_text = ""
        words_info = []

    if ground_truth and ground_truth.strip():
        try:
            metrics = calculate_overall_metrics(ocr_text, ground_truth)
        except Exception as e_metrics:
            print(f"ERROR [extract_text_and_metrics] Error calculating metrics for {image_desc}: {e_metrics}")
            metrics = DEFAULT_METRICS.copy()
    else:
        metrics = DEFAULT_METRICS.copy()

    return ocr_text, words_info, metrics

# --- Preprocessing Function (Unchanged) ---
def preprocess_image(image_cv):
    if image_cv is None:
        print("ERROR [preprocess_image] Input image is None.")
        return None
    try:
        if len(image_cv.shape) < 2 or image_cv.shape[0] <= 0 or image_cv.shape[1] <= 0:
            print(f"ERROR [preprocess_image] Invalid image shape: {image_cv.shape}")
            return image_cv

        gray = image_cv
        if len(image_cv.shape) == 3 and image_cv.shape[2] == 3:
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        elif len(image_cv.shape) == 3 and image_cv.shape[2] == 4:
            bgr_image = cv2.cvtColor(image_cv, cv2.COLOR_BGRA2BGR)
            gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        elif len(image_cv.shape) == 2:
            pass
        else:
            print(f"WARNING [preprocess_image] Unexpected image shape for grayscale conversion: {image_cv.shape}")

        if len(gray.shape) == 2:
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        else:
            print("WARNING [preprocess_image] Skipping Gaussian Blur due to unexpected dimensions.")
            blurred = gray

        if len(image_cv.shape) == 4:
            image_for_ops = cv2.cvtColor(blurred, cv2.COLOR_BGRA2BGR)
        elif len(blurred.shape) == 2:
            print("WARNING [preprocess_image] Input is grayscale, resizing/morphology might not work as expected.")
            image_for_ops = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
        else:
            image_for_ops = blurred

        image_resized = image_for_ops
        try:
            if image_for_ops.shape[0] > 0 and image_for_ops.shape[1] > 0:
                image_resized = cv2.resize(image_for_ops, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
        except Exception as e_resize:
            print(f"ERROR [preprocess_image] Failed during resize: {e_resize}. Using unresized image.")
            image_resized = image_for_ops

        cleaned = image_resized
        try:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            if image_resized is not None and image_resized.shape[0] > 0 and image_resized.shape[1] > 0:
                cleaned = cv2.morphologyEx(image_resized, cv2.MORPH_CLOSE, kernel)
            else:
                print("WARNING [preprocess_image] Skipping Morphological Closing due to invalid input image.")
        except Exception as e_morph:
            print(f"ERROR [preprocess_image] Failed during morphology: {e_morph}. Using resized image.")
            cleaned = image_resized

        return cleaned
    except cv2.error as e_cv:
        print(f"ERROR [preprocess_image] OpenCV Error: {e_cv}")
        return image_cv
    except Exception as e:
        print(f"ERROR [preprocess_image] General Error: {e}")
        return image_cv

# --- Flask Routes (Unchanged) ---
@app.route('/easy-ocr', methods=['POST'])
def upload_image():
    print("\n--- Request Received: /easy-ocr ---")
    if 'image' not in request.files:
        return jsonify({'error': 'No image file found in the request'}), 400
    file = request.files['image']
    print(f"DEBUG: Received file: {file.filename}")

    try:
        image_bytes = np.frombuffer(file.read(), np.uint8)
        original_image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        if original_image is None:
            original_image = cv2.imdecode(image_bytes, cv2.IMREAD_UNCHANGED)
            if original_image is not None:
                if len(original_image.shape) > 2 and original_image.shape[2] == 4:
                    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGRA2BGR)
                elif len(original_image.shape) != 2:
                    print(f"ERROR: Decoded image has unexpected shape: {original_image.shape}")
                    return jsonify({'error': 'Unsupported image format or channel count'}), 400
            else:
                return jsonify({'error': 'Invalid or unsupported image format'}), 400
        print(f"DEBUG: Image decoded successfully. Shape: {original_image.shape}")
    except Exception as e:
        print(f"ERROR: Failed to read/decode image: {e}")
        return jsonify({'error': f'Failed to read or decode image: {str(e)}'}), 400

    test_plan = request.form.get('test_plan', '').lower()
    ground_truth = request.form.get('ground_truth', '').strip()
    print(f"DEBUG: Test Plan: '{test_plan}', Ground Truth Provided: {bool(ground_truth)}")
    if ground_truth: print(f"DEBUG: Ground Truth: '{ground_truth[:100]}...'")

    response = {'test_plan': test_plan, 'ground_truth': ground_truth}

    # Before Preprocessing
    ocr_text_before, words_before, metrics_before = "", [], DEFAULT_METRICS.copy()
    if test_plan == 'test3' and ground_truth:
        print("\nDEBUG: == Running OCR on Original Image ==")
        ocr_text_before, words_before, metrics_before = extract_text_and_metrics(original_image, "Original", ground_truth)

    # Preprocessing
    print("\nDEBUG: == Preprocessing Image ==")
    cleaned_image = preprocess_image(original_image)
    if cleaned_image is None:
        print("ERROR: Preprocessing returned None.")
        return jsonify({'error': 'Image preprocessing failed'}), 500

    # After Preprocessing
    print("\nDEBUG: == Running OCR on Preprocessed Image ==")
    ocr_text_after, words_after, metrics_after = extract_text_and_metrics(
        cleaned_image, "Preprocessed", ground_truth if ground_truth else None
    )

    # Structuring Response
    if test_plan == 'test3' and ground_truth:
        print("\nDEBUG: Structuring response for Test Plan 3 (with GT)...")
        wa_before = metrics_before.get('word_accuracy_ratio_percent')
        wa_after = metrics_after.get('word_accuracy_ratio_percent')
        word_improvement = None

        ca_before = metrics_before.get('char_accuracy_ratio_percent')
        ca_after = metrics_after.get('char_accuracy_ratio_percent')
        char_improvement = None

        if isinstance(wa_before, (int, float)) and isinstance(wa_after, (int, float)):
            word_improvement = round(wa_after - wa_before, 2)
            print(f"DEBUG: Accuracy Improvement (Word Ratio): {word_improvement:.2f}% ...")
        else:
            print(f"DEBUG: Cannot calculate improvement...")

        if isinstance(ca_before, (int, float)) and isinstance(ca_after, (int, float)):
            char_improvement = round(ca_after - ca_before, 2)
            print(f"DEBUG: Accuracy Improvement (Character Ratio): {char_improvement:.2f}% ...")
        else:
            print(f"DEBUG: Cannot calculate improvement...")

        response.update({
            'before_preprocessing': {'ocr_text': ocr_text_before, 'words': words_before, 'metrics': metrics_before},
            'after_preprocessing': {'ocr_text': ocr_text_after, 'words': words_after, 'metrics': metrics_after},
            'word_accuracy_improvement': word_improvement,
            'char_accuracy_improvement': char_improvement,
            'message': 'Evaluation completed successfully.'
        })
    else:
        print("\nDEBUG: Structuring response for other Test Plans or no GT...")
        response.update({
            'after_preprocessing': {'ocr_text': ocr_text_after, 'words': words_after, 'metrics': metrics_after},
            'before_preprocessing': {'ocr_text': ocr_text_before, 'words': words_before, 'metrics': metrics_before},
            'accuracy_improvement': None,
            'message': 'OCR processing complete.' + (' Ground truth evaluation metrics included.' if ground_truth else ' Provide ground truth for evaluation metrics.')
        })

    print(f"\nDEBUG: == Sending Response ==\n{json.dumps(response, indent=2)}\n-----------------------------")
    return jsonify(response)

@app.route('/evaluate-ocr', methods=['POST'])
def evaluate_with_ground_truth():
    print("\n--- Request Received: /evaluate-ocr ---")
    ground_truth = request.form.get('ground_truth', '').strip()
    ocr_text = request.form.get('ocr_text', '')
    print(f"DEBUG: Received GT: '{ground_truth[:80]}...'")
    print(f"DEBUG: Received OCR Text: '{ocr_text[:80]}...'")

    if not ground_truth:
        return jsonify({'error': 'Missing ground truth for evaluation'}), 400

    try:
        metrics = calculate_overall_metrics(ocr_text, ground_truth)
    except Exception as e:
        print(f"ERROR calculating metrics in /evaluate-ocr: {e}")
        metrics = DEFAULT_METRICS.copy()

    response = {'ground_truth': ground_truth, 'ocr_text': ocr_text, 'metrics': metrics}
    print(f"\nDEBUG: == Sending Response ==\n{json.dumps(response, indent=2)}\n-----------------------------")
    return jsonify(response)

if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(host="0.0.0.0", port=5001, debug=True)