# OCR Comparison Hub

The **OCR Comparison Hub** is a Flutter-based application designed to evaluate and compare the performance of various Optical Character Recognition (OCR) models. This tool provides a user-friendly interface to test different OCR engines, including EasyOCR, PaddleOCR, and MMOCR, with various image preprocessing techniques.

---

### 🌟 Features

*   **Multiple OCR Model Integration**: Seamlessly switch between and compare different OCR backends.
    *   OpenCV + EasyOCR
    *   OpenCV + PaddleOCR
    *   OpenCV + MMOCR
*   **Image Preprocessing**: Applies a set of OpenCV techniques to enhance images before they are fed into the OCR models, including grayscaling, Gaussian blur, resizing, and morphological closing.
*   **Performance Evaluation**: Calculates and displays a comprehensive set of metrics to assess the accuracy of the OCR output against ground truth data.
    *   Word and Character Accuracy
    *   Precision, Recall, and F1-Score (at both word and character levels)
    *   Detailed counts of matched and unmatched words/characters.
*   **Side-by-Side Comparison**: For test plans involving ground truth, the application provides a direct comparison of OCR results before and after image preprocessing, highlighting the impact on accuracy.
*   **Cross-Platform Frontend**: Built with Flutter, allowing for a consistent user experience across different platforms.
*   **Flask-Based Backends**: Each OCR model is served via a lightweight Flask server, making the architecture modular and easy to extend.

---

### 🚀 Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

#### Prerequisites

*   Flutter SDK
*   Python 3.8+
*   Conda (recommended for managing Python environments)
*   An IDE such as Visual Studio Code or Android Studio

#### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/ocr-comparison-hub.git
    cd ocr-comparison-hub
    ```

2.  **Set up the Flutter frontend:**
    ```bash
    flutter pub get
    ```

3.  **Set up the Python backends:**
    For each OCR model (easyocr, paddleocr, mmocr), navigate to its respective directory and set up the Python environment. For example, for `mmocr`:
    ```bash
    cd mmocr
    conda create --name openmmlab python=3.8
    conda activate openmmlab
    pip install -r requirements.txt
    ```
    **Note:** The `mmocr` setup may require additional steps for installing PyTorch and MMCV. Please refer to the official MMOCR documentation for detailed instructions.

---

### 🏃‍♀️ Usage

1.  **Run the backend services:**
    For each OCR model, start the Flask server. For example, to run the `mmocr` server:
    ```bash
    conda activate openmmlab
    cd c:/mmocr
    python main.py
    ```
    The server will start on `http://localhost:5003`.

2.  **Run the Flutter application:**
    ```bash
    flutter run
    ```

3.  **Using the App:**
    *   From the main menu, select the OCR method you want to test.
    *   This will navigate you to the respective screen for that OCR model.
    *   On the OCR screen, you can:
        *   Pick an image from your gallery.
        *   Enter the ground truth text for the image.
        *   Select a test plan.
    *   The app will then send the image to the corresponding backend, process it, and display the OCR results and evaluation metrics.

---

### 🏗️ Project Architecture

The project is divided into two main components:

**Frontend (Flutter):**

*   `main.dart`: The entry point of the application, which sets up the main menu.
*   `*_screen.dart`: Separate screens for each OCR model (`easy_ocr_screen.dart`, `paddle_ocr_screen.dart`, `mmocr_screen.dart`). These screens handle user interaction, image selection, and communication with the backend services.

**Backend (Python + Flask):**

*   Each OCR model resides in its own directory (`easyocr`, `paddleocr`, `mmocr`) and runs as a separate Flask server.
*   `main.py` (within each model's directory): This file contains the core logic for the backend service, including:
    *   A Flask API endpoint (e.g., `/mmocr`) to receive image data.
    *   **Image Preprocessing**: A `preprocess_image` function to clean and enhance the input image.
    *   **OCR Text Extraction**: An `extract_text_and_metrics` function that uses the specific OCR library (e.g., mmocr-inferencer) to perform text detection and recognition.
    *   **Metric Calculation**: A `calculate_overall_metrics` function to compute various accuracy metrics based on the OCR output and the provided ground truth. This uses `difflib` for comparing sequences.
*   **Models and Weights**: The `weights` directory for each model contains the pre-trained model files required for the OCR inference.

---

### 📊 Evaluation Metrics

The application uses a comprehensive set of metrics to evaluate the performance of the OCR models:

*   **Accuracy Ratios**: Calculated using `difflib.SequenceMatcher`, these provide a percentage of similarity between the OCR output and the ground truth at both the word and character levels.
*   **Precision, Recall, and F1-Score**: These metrics provide a more nuanced understanding of the OCR performance by considering true positives, false positives, and false negatives. They are also calculated at both the word and character levels.
*   **Counts**: The application also provides raw counts of total words and characters, matched words and characters, and unmatched words and characters for both the ground truth and the OCR output.

---

### ✅ To-Do

- [ ] Implement the `EasyOCRScreen` and `PaddleOCRScreen` in the Flutter application.
- [ ] Add support for more OCR models.
- [ ] Allow for customizable image preprocessing pipelines.
- [ ] Implement batch processing of images for more extensive testing.
- [ ] Visualize the bounding boxes of the detected text on the original image.
- [ ] Improve error handling and user feedback in the application.
