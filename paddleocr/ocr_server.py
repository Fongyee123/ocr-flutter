from flask import Flask, request, jsonify
import cv2
import numpy as np
from paddleocr import PaddleOCR # Keep PaddleOCR
import paddle                 # Keep Paddle specific imports
from flask_cors import CORS
import re
import difflib
import json

# Force PaddleOCR to use CPU
paddle.set_device('cpu') # Keep Paddle setting

app = Flask(__name__)
CORS(app)

# Initialize OCR (PaddleOCR)
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False) # Keep PaddleOCR initialization

def normalize_text(text):
    """Removes punctuation, converts to lowercase, handles extra spaces."""
    if not isinstance(text, str): # Basic type check
        text = str(text)
    text_no_punct = re.sub(r'[^\w\s]', '', text)
    text_single_space = re.sub(r'\s+', ' ', text_no_punct).strip()
    lower_text = text_single_space.lower()
    return lower_text

# --- Accuracy/Ratio Functions (Removed ordered versions) ---
def word_accuracy(ocr_text, ground_truth):
    """Calculates word-level similarity ratio (order-insensitive)."""
    ocr_words = normalize_text(ocr_text).split()
    gt_words = normalize_text(ground_truth).split()
    if not gt_words and not ocr_words: return 1.0
    if not gt_words or not ocr_words: return 0.0
    ratio = difflib.SequenceMatcher(None, ocr_words, gt_words).ratio()
    return ratio

def char_accuracy(ocr_text, ground_truth):
    """Calculates character-level similarity ratio (order-insensitive)."""
    ocr_text_norm = normalize_text(ocr_text)
    ground_truth_norm = normalize_text(ground_truth)
    if not ground_truth_norm and not ocr_text_norm: return 1.0
    if not ground_truth_norm or not ocr_text_norm: return 0.0
    ratio = difflib.SequenceMatcher(None, ocr_text_norm, ground_truth_norm).ratio()
    return ratio

# --- REMOVED word_accuracy_ordered ---
# --- REMOVED char_accuracy_ordered ---

# --- Function to calculate ALL metrics, including new counts ---

def calculate_overall_metrics(ocr_text, ground_truth):
    """Calculates all metrics including detailed counts, matches, mismatches, and character PRF."""
    # Normalize texts first
    ocr_text_norm = normalize_text(ocr_text)
    gt_text_norm = normalize_text(ground_truth)
    ocr_words_norm = ocr_text_norm.split()
    gt_words_norm = gt_text_norm.split()

    # --- 1. Basic Counts ---
    total_ocr_words = len(ocr_words_norm)
    total_ocr_chars = len(ocr_text_norm)
    total_gt_words = len(gt_words_norm)
    total_gt_chars = len(gt_text_norm)

    metrics = {
        # Counts
        'total_ocr_words': total_ocr_words,
        'total_ocr_chars': total_ocr_chars,
        'total_gt_words': total_gt_words,
        'total_gt_chars': total_gt_chars,
    }

    # --- 2. Matching and Error Calculation (using SequenceMatcher) ---
    matched_words = 0
    matched_chars = 0
    word_errors = 0
    char_errors = 0
    ocr_unmatched_words = 0
    ocr_unmatched_chars = 0
    gt_unmatched_words = 0
    gt_unmatched_chars = 0

    # Calculate word matches/errors
    if total_gt_words > 0 or total_ocr_words > 0:
        matcher_w = difflib.SequenceMatcher(None, ocr_words_norm, gt_words_norm)
        matched_words = sum(block.size for block in matcher_w.get_matching_blocks())
        word_errors = total_ocr_words + total_gt_words - 2 * matched_words
        ocr_unmatched_words = total_ocr_words - matched_words
        gt_unmatched_words = total_gt_words - matched_words

    # Calculate character matches/errors
    if total_gt_chars > 0 or total_ocr_chars > 0:
        matcher_c = difflib.SequenceMatcher(None, ocr_text_norm, gt_text_norm)
        matched_chars = sum(block.size for block in matcher_c.get_matching_blocks())
        char_errors = total_ocr_chars + total_gt_chars - 2 * matched_chars
        ocr_unmatched_chars = total_ocr_chars - matched_chars
        gt_unmatched_chars = total_gt_chars - matched_chars

    metrics.update({
        # Matches (based on SequenceMatcher)
        'matched_words': matched_words,
        'matched_chars': matched_chars,
        # Unmatched (relative to GT - how many GT items were missed/wrong)
        'gt_unmatched_words': gt_unmatched_words,
        'gt_unmatched_chars': gt_unmatched_chars,
        # Unmatched (relative to OCR - how many OCR items were extra/wrong)
        'ocr_unmatched_words': ocr_unmatched_words, # Often considered False Positives
        'ocr_unmatched_chars': ocr_unmatched_chars, # Often considered False Positives
         # Errors (Edit Distance)
        'word_errors': word_errors, # This is Ins+Del+Sub
        'char_errors': char_errors, # This is Ins+Del+Sub
    })

    # --- 3. Accuracy Ratios / Percentages (Removed ordered versions) ---
    metrics.update({
        'word_accuracy_ratio_percent': round(word_accuracy(ocr_text, ground_truth) * 100, 2),
        'char_accuracy_ratio_percent': round(char_accuracy(ocr_text, ground_truth) * 100, 2),
        # 'word_accuracy_ordered_percent': REMOVED
        # 'char_accuracy_ordered_percent': REMOVED
    })

    # --- 4. Precision, Recall, F1 (Word Level) ---
    tp_w = matched_words
    fp_w = ocr_unmatched_words # Words in OCR not matching GT
    fn_w = gt_unmatched_words  # Words in GT not matching OCR

    precision_w = tp_w / (tp_w + fp_w) if (tp_w + fp_w) > 0 else (1.0 if total_gt_words == 0 and total_ocr_words == 0 else 0.0)
    recall_w = tp_w / (tp_w + fn_w) if (tp_w + fn_w) > 0 else (1.0 if total_gt_words == 0 and total_ocr_words == 0 else 0.0)
    f1_w = 2 * (precision_w * recall_w) / (precision_w + recall_w) if (precision_w + recall_w) > 0 else (1.0 if total_gt_words == 0 and total_ocr_words == 0 else 0.0)

    metrics.update({
        'precision': round(precision_w * 100, 2),
        'recall': round(recall_w * 100, 2),
        'f1_score': round(f1_w * 100, 2)
    })

    # --- 5. Precision, Recall, F1 (Character Level) - ADDED ---
    tp_c = matched_chars
    fp_c = ocr_unmatched_chars # Characters in OCR not matching GT
    fn_c = gt_unmatched_chars  # Characters in GT not matching OCR

    precision_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else (1.0 if total_gt_chars == 0 and total_ocr_chars == 0 else 0.0)
    recall_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else (1.0 if total_gt_chars == 0 and total_ocr_chars == 0 else 0.0)
    f1_c = 2 * (precision_c * recall_c) / (precision_c + recall_c) if (precision_c + recall_c) > 0 else (1.0 if total_gt_chars == 0 and total_ocr_chars == 0 else 0.0)

    metrics.update({
        'precision_c': round(precision_c * 100, 2),
        'recall_c': round(recall_c * 100, 2),
        'f1_score_c': round(f1_c * 100, 2)
    })

    return metrics


# --- Update Default Metrics Structure ---
# Reflects the removed ordered accuracy and added character PRF
DEFAULT_METRICS = {
    # Counts
    'total_ocr_words': None, 'total_ocr_chars': None,
    'total_gt_words': None, 'total_gt_chars': None,
    # Matches
    'matched_words': None, 'matched_chars': None,
    # Unmatched
    'gt_unmatched_words': None, 'gt_unmatched_chars': None,
    'ocr_unmatched_words': None, 'ocr_unmatched_chars': None,
    # Errors (Edit Distance)
    'word_errors': None, 'char_errors': None,
    # Accuracy Ratios/Percentages
    'word_accuracy_ratio_percent': None, 'char_accuracy_ratio_percent': None,
    # REMOVED: 'word_accuracy_ordered_percent': None,
    # REMOVED: 'char_accuracy_ordered_percent': None,
    # PRF (Word)
    'precision': None, 'recall': None, 'f1_score': None,
    # PRF (Character) - ADDED
    'precision_c': None, 'recall_c': None, 'f1_score_c': None
}


# --- Update extract_text_and_metrics to use the new default and PaddleOCR ---
def extract_text_and_metrics(image, image_desc="image", ground_truth=None):
    """Extracts text using PaddleOCR and calculates metrics if ground truth is provided."""
    ocr_text, words_info, metrics = "", [], {} # Initialize defaults
    try:
        # USE PADDLEOCR HERE
        result = ocr.ocr(image, cls=True)
        all_text = []
        words_info = []

        # Keep the robust PaddleOCR result parsing logic
        if result is not None and result[0] is not None:
             for line_result in result[0]:
                if isinstance(line_result, list) and len(line_result) == 2 and isinstance(line_result[0], list) and len(line_result[0]) == 4 and isinstance(line_result[1], (tuple, list)) and len(line_result[1]) == 2:
                    box = line_result[0]
                    text_info = line_result[1]
                    if all(isinstance(pt, list) and len(pt) == 2 and all(isinstance(coord, (int, float)) for coord in pt) for pt in box):
                        try:
                            int_box = [[int(coord) for coord in pt] for pt in box]
                            (x1, y1), (x2, y2), (x3, y3), (x4, y4) = int_box
                        except (ValueError, TypeError): continue
                    else: continue

                    text = text_info[0]
                    conf = text_info[1]
                    if not isinstance(text, str): continue
                    try:
                        conf_float = round(float(conf) * 100, 2)
                    except (ValueError, TypeError): continue

                    if conf_float >= 50: # Confidence threshold
                        all_text.append(text)
                        words_info.append({
                            'text': text, 'confidence': conf_float,
                            'bounding_box': {'top_left': [x1, y1], 'top_right': [x2, y2], 'bottom_right': [x3, y3], 'bottom_left': [x4, y4]}
                        })
             ocr_text = ' '.join(all_text)
        else:
             ocr_text = ""

    except Exception as e:
        print(f"ERROR [extract_text_and_metrics] Error during PaddleOCR for {image_desc}: {e}") # Updated error message
        ocr_text = ""
        words_info = []

    # --- Use the updated DEFAULT_METRICS structure ---
    if ground_truth and ground_truth.strip():
        try:
            # Call the enhanced calculate_overall_metrics function
            metrics = calculate_overall_metrics(ocr_text, ground_truth)
        except Exception as e_metrics:
            print(f"ERROR [extract_text_and_metrics] Error calculating metrics for {image_desc}: {e_metrics}")
            metrics = DEFAULT_METRICS.copy()
    else:
        metrics = DEFAULT_METRICS.copy()

    return ocr_text, words_info, metrics

# --- Preprocessing Function (Unchanged from your original PaddleOCR version) ---
def preprocess_image(image_cv):
    # ... (Keep the preprocess_image function from the previous version) ...
    # print("\nDEBUG [preprocess_image] Starting...") # Keep debug optional
    if image_cv is None:
        print("ERROR [preprocess_image] Input image is None.")
        return None
    try:
        # Check if image has valid dimensions
        if len(image_cv.shape) < 2 or image_cv.shape[0] <= 0 or image_cv.shape[1] <= 0:
             print(f"ERROR [preprocess_image] Invalid image shape: {image_cv.shape}")
             return image_cv # Return original if shape is invalid

        # Grayscale conversion only if it's a color image
        gray = image_cv
        if len(image_cv.shape) == 3 and image_cv.shape[2] == 3:
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            # print("DEBUG [preprocess_image] Converted to grayscale.") # Keep debug optional
        elif len(image_cv.shape) == 3 and image_cv.shape[2] == 4:
             bgr_image = cv2.cvtColor(image_cv, cv2.COLOR_BGRA2BGR)
             gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
             # print("DEBUG [preprocess_image] Converted BGRA to grayscale.") # Keep debug optional
        elif len(image_cv.shape) == 2:
             # print("DEBUG [preprocess_image] Image is already grayscale.") # Keep debug optional
             pass # Already grayscale
        else:
             print(f"WARNING [preprocess_image] Unexpected image shape for grayscale conversion: {image_cv.shape}")

        # Gaussian Blur (ensure input is 2D)
        if len(gray.shape) == 2:
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            # print("DEBUG [preprocess_image] Applied Gaussian Blur (5x5).") # Keep debug optional
        else:
            print("WARNING [preprocess_image] Skipping Gaussian Blur due to unexpected dimensions after grayscale attempt.")
            blurred = gray # Fallback

        # Ensure original_image is BGR for resizing and morphology
        if len(blurred.shape) == 4: # Assuming BGRA input originally
            image_for_ops = cv2.cvtColor(blurred, cv2.COLOR_BGRA2BGR)
        elif len(blurred.shape) == 2:
            print("WARNING [preprocess_image] Input is grayscale, resizing/morphology might not work as expected.")
            image_for_ops = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR) # Convert for function calls
        else:
             image_for_ops = image_cv # Assume BGR

        # Resize (Original Color, now ensuring it's 3-channel)
        image_resized = image_for_ops
        try:
             if image_for_ops.shape[0] > 0 and image_for_ops.shape[1] > 0:
                 image_resized = cv2.resize(image_for_ops, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
                 # print(f"DEBUG [preprocess_image] Resized image by 1.2x to {image_resized.shape[:2]}.") # Keep debug optional
        except Exception as e_resize:
             print(f"ERROR [preprocess_image] Failed during resize: {e_resize}. Using unresized image.")
             image_resized = image_for_ops # Fallback

        # Morphological Closing (Resized Color)
        cleaned = image_resized
        try:
             kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
             if image_resized is not None and image_resized.shape[0] > 0 and image_resized.shape[1] > 0:
                 cleaned = cv2.morphologyEx(image_resized, cv2.MORPH_CLOSE, kernel)
                 # print("DEBUG [preprocess_image] Applied Morphological Closing (3x3 kernel) on resized color image.") # Keep debug optional
             else:
                  print("WARNING [preprocess_image] Skipping Morphological Closing due to invalid input image.")
        except Exception as e_morph:
            print(f"ERROR [preprocess_image] Failed during morphology: {e_morph}. Using resized image.")
            cleaned = image_resized # Fallback

        # print("DEBUG [preprocess_image] Finished.") # Keep debug optional
        return cleaned
    except cv2.error as e_cv:
        print(f"ERROR [preprocess_image] OpenCV Error: {e_cv}")
        return image_cv # Return original on OpenCV error
    except Exception as e:
        print(f"ERROR [preprocess_image] General Error: {e}")
        return image_cv # Return original on general error


# --- Flask Routes (Updated Improvement Calculation) ---

@app.route('/paddle-ocr', methods=['POST'])
def upload_image():
    print("\n--- Request Received: /paddle-ocr ---")
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

    # --- Before Preprocessing ---
    ocr_text_before, words_before, metrics_before = "", [], DEFAULT_METRICS.copy()
    if test_plan == 'test3' and ground_truth:
        print("\nDEBUG: == Running OCR on Original Image ==")
        # Use the updated extract_text_and_metrics (which uses PaddleOCR)
        ocr_text_before, words_before, metrics_before = extract_text_and_metrics(original_image, "Original", ground_truth)

    # --- Preprocessing ---
    print("\nDEBUG: == Preprocessing Image ==")
    cleaned_image = preprocess_image(original_image)
    if cleaned_image is None:
        print("ERROR: Preprocessing returned None.")
        return jsonify({'error': 'Image preprocessing failed'}), 500

    # --- After Preprocessing ---
    print("\nDEBUG: == Running OCR on Preprocessed Image ==")
    # Use the updated extract_text_and_metrics (which uses PaddleOCR)
    ocr_text_after, words_after, metrics_after = extract_text_and_metrics(
        cleaned_image, "Preprocessed", ground_truth if ground_truth else None
    )

    # --- Structuring Response (Updated Improvement Calculation) ---
    if test_plan == 'test3' and ground_truth:
        print("\nDEBUG: Structuring response for Test Plan 3 (with GT)...")
        # Calculate improvement based on word and char accuracy ratios
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
             print(f"DEBUG: Cannot calculate word accuracy improvement...")

        if isinstance(ca_before, (int, float)) and isinstance(ca_after, (int, float)):
             char_improvement = round(ca_after - ca_before, 2)
             print(f"DEBUG: Accuracy Improvement (Character Ratio): {char_improvement:.2f}% ...")
        else:
             print(f"DEBUG: Cannot calculate char accuracy improvement...")

        response.update({
            'before_preprocessing': {'ocr_text': ocr_text_before, 'words': words_before, 'metrics': metrics_before},
            'after_preprocessing': {'ocr_text': ocr_text_after, 'words': words_after, 'metrics': metrics_after},
            'word_accuracy_improvement': word_improvement, # Use new key
            'char_accuracy_improvement': char_improvement, # Use new key
            'message': 'Evaluation completed successfully.'
        })
    else:
        print("\nDEBUG: Structuring response for other Test Plans or no GT...")
        response.update({
            'after_preprocessing': {'ocr_text': ocr_text_after, 'words': words_after, 'metrics': metrics_after},
            'before_preprocessing': {'ocr_text': ocr_text_before, 'words': words_before, 'metrics': metrics_before},
            'word_accuracy_improvement': None, # Include keys for consistency
            'char_accuracy_improvement': None, # Include keys for consistency
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
        # This now calls the updated calculate_overall_metrics
        metrics = calculate_overall_metrics(ocr_text, ground_truth)
    except Exception as e:
        print(f"ERROR calculating metrics in /evaluate-ocr: {e}")
        metrics = DEFAULT_METRICS.copy() # Use updated default on error

    response = {'ground_truth': ground_truth, 'ocr_text': ocr_text, 'metrics': metrics}
    print(f"\nDEBUG: == Sending Response ==\n{json.dumps(response, indent=2)}\n-----------------------------")
    return jsonify(response)

# --- End of Flask Routes ---

if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(host="0.0.0.0", port=5002, debug=True) # Set debug=False for production