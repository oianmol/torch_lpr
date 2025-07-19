import os
import cv2
import numpy as np
import base64
import datetime
import logging
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- TensorFlow Imports (Keep for TF model support if desired) ---
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- PyTorch Imports ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF # Import functional for direct use

from easyocr import Reader
import warnings

from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Tuple, Optional

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore')  # General warnings filter

# Configure GPU growth for TensorFlow if available
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logging.info(f"✅ Configured {len(gpus)} GPU(s) for dynamic memory growth (TensorFlow)")
except Exception as e:
    logging.warning(f"⚠️ TensorFlow GPU configuration failed: {e}")

# Determine PyTorch device
PYTORCH_DEVICE = torch.device("mps")
logging.info(f"✅ PyTorch will use device: {PYTORCH_DEVICE}")


@dataclass
class ProcessingConfig:
    """Configuration class for processing parameters"""
    # Directories
    image_dir: str = "plates_org/test/"
    model_path: str = "emnist_cnn.h5"  # Could be .h5 or .pth
    output_dir: str = "output/"
    debug_dir: str = "debug_chars/"
    num_classes: int = 0 # This should be set by the model loading now
    # Processing parameters
    batch_size: int = 32
    confidence_threshold: float = 0.7
    max_workers: int = 4



    # Image preprocessing
    adaptive_preprocessing: bool = True
    perspective_correction: bool = True
    noise_reduction: bool = True

    # Character filtering
    min_char_width: int = 8
    min_char_height: int = 12
    min_char_height_ratio: float = 0.1
    max_char_height_ratio: float = 0.8
    min_aspect_ratio: float = 0.15
    max_aspect_ratio: float = 1.2
    min_char_area: int = 50

    # OCR settings
    ocr_languages: List[str] = None
    ocr_gpu: bool = True

    # Output settings
    generate_html: bool = True
    save_debug_images: bool = True
    save_contour_images: bool = True # NEW: Option to save contour highlighted images


    def __post_init__(self):
        if self.ocr_languages is None:
            self.ocr_languages = ['en']


# --- Define PyTorch EMNIST CNN Model (Example) ---
class EMNISTCNN(nn.Module):
    def __init__(self, num_classes):
        super(EMNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# --- EMNIST Label Mapping ---
EMNIST_62_LABEL_MAP = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I',
    19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R',
    28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',
    36: 'a', 37: 'b', 38: 'c', 39: 'd', 40: 'e', 41: 'f', 42: 'g', 43: 'h', 44: 'i',
    45: 'j', 46: 'k', 47: 'l', 48: 'm', 49: 'n', 50: 'o', 51: 'p', 52: 'q', 53: 'r',
    54: 's', 55: 't', 56: 'u', 57: 'v', 58: 'w', 59: 'x', 60: 'y', 61: 'z'
}

EMNIST_BALANCED_LABEL_MAP = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I',
    19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R',
    28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',
    36: 'a', 37: 'b', 38: 'd', 39: 'e', 40: 'f', 41: 'g', 42: 'h', 43: 'n',
    44: 'q', 45: 'r', 46: 't'
}

EMNIST_BALANCED_CUSTOM_LABEL_MAP = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I',
    19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R',
    28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z'
}


class OptimizedPlateRecognizer:
    """Optimized license plate recognition system"""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.yolo_model = YOLO("yolo11n-cls.pt")  # Use your trained YOLOv8 model
        #results = self.yolo_model.train(data="mnist", epochs=100, imgsz=32)
        self.logger = self._setup_logging()
        self.model = None
        self.model_type = "UNKNOWN"
        self.num_classes = config.num_classes # Initialized from config, updated by _load_model
        self.label_map = {}  # Store the correct label map based on model type
        self.ocr_reader = None
        self.pytorch_transform = transforms.Compose([
            transforms.ToTensor(),  # Converts numpy array (H, W, C) to (C, H, W) and normalizes to [0, 1]
            transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
        ])

        # Create output directories
        self._create_directories()

        # Load model and OCR reader
        self._load_model()
        self._initialize_ocr()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        log_dir = Path(self.config.output_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"plate_recognition_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ],
            force=True
        )

        logger = logging.getLogger(__name__)
        logger.info(f"✅ Logging initialized. Log file: {log_file}")
        return logger

    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.config.output_dir,
            self.config.debug_dir,
            Path(self.config.output_dir) / "reports",
            Path(self.config.output_dir) / "json",
            Path(self.config.output_dir) / "contour_images" # NEW: Directory for contour images
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def _load_model(self):
        """Load and analyze the model (supports Keras .h5 and PyTorch .pth)"""
        if not os.path.exists(self.config.model_path):
            self.logger.warning(f"⚠️ Model file not found: {self.config.model_path}")
            return

        model_ext = Path(self.config.model_path).suffix.lower()

        try:
            if model_ext == '.h5':
                self.model = load_model(self.config.model_path)
                output_layer = self.model.layers[-1]
                self.num_classes = output_layer.output.shape[-1]
                self.model_backend = "keras"
                self.logger.info(f"✅ Keras model loaded from {self.config.model_path}")

            elif model_ext == '.pth':
                if self.config.num_classes == 0: # Ensure num_classes is set for PyTorch model instantiation
                    raise ValueError("config.num_classes must be set for PyTorch models (e.g., 47 for EMNIST_Balanced, 62 for EMNIST_ByClass).")
                self.num_classes = self.config.num_classes # Use config's num_classes for PyTorch
                self.model = EMNISTCNN(self.num_classes)
                self.model.load_state_dict(torch.load(self.config.model_path, map_location=PYTORCH_DEVICE))
                self.model.eval()  # Set to evaluation mode
                self.model.to(PYTORCH_DEVICE)
                self.model_backend = "pytorch"
                self.logger.info(f"✅ PyTorch model loaded from {self.config.model_path} to {PYTORCH_DEVICE}")

            else:
                self.logger.error(f"❌ Unsupported model file extension: {model_ext}")
                self.model = None
                return

            # Determine model type (based on num_classes)
            if self.num_classes == 10:
                self.model_type = "MNIST"
                self.logger.info("✅ MNIST model loaded (digits 0-9 only)")
                self.logger.warning("⚠️ Note: This model can only predict digits 0-9, not letters")
                self.label_map = {i: str(i) for i in range(10)}
            elif self.num_classes == 47:
                self.model_type = "EMNIST_BALANCED"
                self.logger.info("✅ EMNIST Balanced model loaded (47 classes: digits, merged letters)")
                self.label_map = EMNIST_BALANCED_LABEL_MAP
            elif self.num_classes == 36:
                self.model_type = "EMNIST_BALANCED_CUSTOM"
                self.logger.info("✅ EMNIST Balanced custom model loaded (36 classes: digits, merged letters)")
                self.label_map = EMNIST_BALANCED_CUSTOM_LABEL_MAP
            elif self.num_classes == 62:
                self.model_type = "EMNIST_BYCLASS"
                self.logger.info("✅ EMNIST ByClass model loaded (62 classes: digits, upper A-Z, lower a-z)")
                self.label_map = EMNIST_62_LABEL_MAP
            else:
                self.model_type = "UNKNOWN"
                self.logger.warning(f"⚠️ Unknown model type with {self.num_classes} output classes")
                self.label_map = {i: f"CLASS_{i}" for i in range(self.num_classes)}  # Fallback map

        except Exception as e:
            self.logger.error(f"❌ Could not load model {self.config.model_path}: {e}")
            self.model = None
            self.model_backend = "NONE"

    def _initialize_ocr(self):
        """Initialize OCR reader"""
        try:
            self.ocr_reader = Reader(self.config.ocr_languages, gpu=self.config.ocr_gpu)
            self.logger.info(f"✅ EasyOCR initialized with languages: {self.config.ocr_languages}")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize OCR: {e}")
            self.ocr_reader = None

    def _get_label_from_idx(self, idx: int) -> str:
        """Map model indices to characters using the chosen label map"""
        return self.label_map.get(idx, f"UNKNOWN_{idx}")

    def predict_with_confidence_calibration(self, char_images_list: List[np.ndarray]) -> Tuple[List[str], float]:
        """Predict characters with confidence calibration for Keras or PyTorch model"""
        if self.model is None or not char_images_list:
            return [], 0.0

        try:
            if self.model_backend == "keras":
                char_images_np = np.array(char_images_list).reshape(-1, 28, 28, 1).astype("float32") / 255.0
                predictions = self.model.predict(char_images_np, verbose=0)

                temperature = 1.5 if self.model_type.startswith("EMNIST") else 1.0
                calibrated_preds = predictions / temperature
                calibrated_preds = tf.nn.softmax(calibrated_preds).numpy()

            elif self.model_backend == "pytorch":
                char_tensors = [self.pytorch_transform(img).float() for img in char_images_list]
                batch_tensors = torch.stack(char_tensors).to(PYTORCH_DEVICE)

                with torch.no_grad():
                    outputs = self.model(batch_tensors)

                predictions = F.softmax(outputs, dim=1).cpu().numpy()
                calibrated_preds = predictions

            else:
                self.logger.error(f"❌ Unknown model backend: {self.model_backend}")
                return [], 0.0

            labels = [self._get_label_from_idx(np.argmax(pred)) for pred in calibrated_preds]
            confidence = float(np.mean([np.max(pred) for pred in calibrated_preds]))

            return labels, confidence

        except Exception as e:
            self.logger.error(f"❌ Model prediction failed: {e}")
            return [], 0.0

    def segment_digits_opencv(self, image_path: str) -> Tuple[
        List[np.ndarray], List[Tuple[int, int, int, int]], Optional[np.ndarray]]:
        """
        Improved OpenCV-based digit segmentation using contour detection.
        Returns:
            - List of 28x28 grayscale digit images
            - List of bounding boxes (x, y, w, h)
            - Annotated image with bounding boxes
        """
        image = cv2.imread(image_path)
        if image is None:
            self.logger.error(f"❌ Could not load image: {image_path}")
            return [], [], None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 31, 15
        )

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        digit_boxes = []
        digit_images = []
        annotated = image.copy()

        h_img, w_img = gray.shape
        i = 0
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            aspect = w / float(h)
            # Filter by area, aspect ratio, and height ratio
            if (
                    area > self.config.min_char_area and
                    self.config.min_char_height_ratio * h_img < h < self.config.max_char_height_ratio * h_img and
                    self.config.min_aspect_ratio < aspect < self.config.max_aspect_ratio and
                    w >= self.config.min_char_width and
                    h >= self.config.min_char_height
            ):
                digit_crop = thresh[y:y + h, x:x + w]
                # Square padding
                size = max(w, h)
                padded = np.zeros((size, size), dtype=np.uint8)
                x_offset = (size - w) // 2
                y_offset = (size - h) // 2
                padded[y_offset:y_offset + h, x_offset:x_offset + w] = digit_crop
                resized = cv2.resize(padded, (28, 28), interpolation=cv2.INTER_AREA)
                digit_images.append(resized)
                if self.config.save_debug_images:
                    debug_path = Path(self.config.debug_dir) / f"{Path(image_path).stem}_char_{i}.png"
                    cv2.imwrite(str(debug_path), resized)
                digit_boxes.append((x, y, w, h))
                i += 1
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Sort digits left-to-right
        digit_boxes, digit_images = zip(
            *sorted(zip(digit_boxes, digit_images), key=lambda b: b[0][0])) if digit_images else ([], [])

        return list(digit_images), list(digit_boxes), annotated

    def process_single_image(self, image_path: str) -> Dict[str, Any]:
        """Process a single image and return results with both model and OCR predictions"""
        try:
            filename = Path(image_path).name
            contour_img_b64 = ""

            char_images, char_boxes, highlighted_img_np = self.segment_digits_opencv(image_path)

            if highlighted_img_np is not None and self.config.save_contour_images:
                _, img_encoded = cv2.imencode('.png', highlighted_img_np)
                contour_img_b64 = f"data:image/png;base64,{base64.b64encode(img_encoded.tobytes()).decode('utf-8')}"

            # Model prediction
            model_prediction = ""
            model_confidence = 0.0
            model_method = ""
            if self.model is not None and char_images:
                labels, conf = self.predict_with_confidence_calibration(char_images)
                model_prediction = self.apply_license_plate_rules(''.join(labels))
                model_confidence = conf
                model_method = f"Model ({self.model_backend})"

            # OCR prediction
            ocr_prediction = ""
            ocr_confidence = 0.0
            ocr_method = ""
            if self.ocr_reader:
                try:
                    ocr_results = self.ocr_reader.readtext(image_path)
                    if ocr_results:
                        ocr_results_sorted = sorted(ocr_results, key=lambda r: r[2], reverse=True)
                        ocr_text = ocr_results_sorted[0][1]
                        ocr_conf = float(ocr_results_sorted[0][2])
                        ocr_prediction = self.apply_license_plate_rules(ocr_text)
                        ocr_confidence = ocr_conf
                        ocr_method = "EasyOCR"
                except Exception as e:
                    self.logger.warning(f"⚠️ OCR failed for {filename}: {e}")

            # Choose final prediction (keep your logic or just pick the higher confidence)
            if model_confidence >= ocr_confidence:
                prediction = model_prediction
                confidence = model_confidence
                method = model_method
            else:
                prediction = ocr_prediction
                confidence = ocr_confidence
                method = ocr_method

            img_b64 = self._image_to_base64(image_path)

            result = {
                "filename": filename,
                "full_path": image_path,
                "prediction": prediction if prediction else "NO_PREDICTION",
                "confidence": confidence,
                "method": method,
                "img_b64": img_b64,
                "contour_img_b64": contour_img_b64,
                "folder": str(Path(image_path).parent.relative_to(self.config.image_dir)),
                "model_prediction": model_prediction,
                "model_confidence": model_confidence,
                "model_method": model_method,
                "ocr_prediction": ocr_prediction,
                "ocr_confidence": ocr_confidence,
                "ocr_method": ocr_method,
            }

            self.logger.debug(f"✅ Processed {filename}: {prediction} (conf: {confidence:.3f}, method: {method})")
            return result

        except Exception as e:
            self.logger.error(f"❌ Failed to process {image_path}: {e}")
            return {
                "filename": Path(image_path).name if image_path else "unknown",
                "full_path": image_path,
                "prediction": "ERROR",
                "confidence": 0.0,
                "method": "Failed",
                "img_b64": "",
                "contour_img_b64": "",
                "folder": "unknown",
                "model_prediction": "",
                "model_confidence": 0.0,
                "model_method": "",
                "ocr_prediction": "",
                "ocr_confidence": 0.0,
                "ocr_method": "",
            }


    def process_images_batched(self) -> Dict[str, Any]:
        """Process images in batches with multithreading"""
        self.logger.info(f"🔍 Processing images from {self.config.image_dir}")

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []

        for root, _, files in os.walk(self.config.image_dir):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    image_files.append(os.path.join(root, file))

        if not image_files:
            self.logger.warning(f"⚠️ No images found in {self.config.image_dir}")
            return {"by_folder": {}, "low_confidence": []}

        self.logger.info(f"📊 Found {len(image_files)} images to process")

        all_results = []

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_path = {executor.submit(self.process_single_image, path): path
                              for path in image_files}

            for future in as_completed(future_to_path):
                try:
                    result = future.result()
                    all_results.append(result)
                except Exception as e:
                    path = future_to_path[future]
                    self.logger.error(f"❌ Failed to process {path}: {e}")

        results_by_folder = {}
        low_confidence_results = []

        for result in all_results:
            folder = result["folder"]

            if folder not in results_by_folder:
                results_by_folder[folder] = []
            results_by_folder[folder].append(result)

            if result["confidence"] < self.config.confidence_threshold:
                low_confidence_results.append(result)

        for folder in results_by_folder:
            results_by_folder[folder].sort(key=lambda x: x["confidence"])

        low_confidence_results.sort(key=lambda x: x["confidence"])

        return {
            "by_folder": results_by_folder,
            "low_confidence": low_confidence_results
        }

    def apply_license_plate_rules(self, prediction: str) -> str:
        """
        Corrects common OCR errors in Indian license plate predictions and applies format rules.
        """
        dict_char_to_int = {'O': '0', 'D': '0', 'Q': '0', 'T': '1', 'Z': '2', 'J': '3', 'A': '4', 'S': '5', 'G': '6',
                            'K': '7', 'B': '8'}
        dict_int_to_char = {'0': 'Q', '1': 'T', '2': 'Z', '3': 'J', '4': 'A', '5': 'S', '6': 'G', '7': 'T', '8': 'B'}
        dict_no_change = {'O': '0', 'I': '1'}

        def preprocess_plate_number(plate_number):
            cleaned_plate = plate_number.replace(" ", "").upper()
            if len(cleaned_plate) >= 4 and cleaned_plate[2:4] == "VA":
                return preprocess_vintage_series(cleaned_plate)
            elif len(cleaned_plate) >= 4 and cleaned_plate[2:4] == "BH":
                return preprocess_bharat_series(cleaned_plate)
            else:
                return preprocess_standard_series(cleaned_plate)

        def preprocess_standard_series(plate_number):
            if len(plate_number) < 7:
                return plate_number
            corrected = list(plate_number)
            for i in range(2):
                if i < len(corrected):
                    char = corrected[i]
                    if char.isdigit() and char in dict_int_to_char:
                        corrected[i] = dict_int_to_char[char]
                    elif char in dict_no_change:
                        corrected[i] = dict_no_change[char]
            for i in range(2, 4):
                if i < len(corrected):
                    char = corrected[i]
                    if char.isalpha() and char in dict_char_to_int:
                        corrected[i] = dict_char_to_int[char]
                    elif char in dict_no_change:
                        corrected[i] = dict_no_change[char]
            series_end = len(corrected) - 4
            for i in range(4, series_end):
                if i < len(corrected):
                    char = corrected[i]
                    if char.isdigit() and char in dict_int_to_char:
                        corrected[i] = dict_int_to_char[char]
                    elif char in dict_no_change:
                        corrected[i] = dict_no_change[char]
            for i in range(len(corrected) - 4, len(corrected)):
                if i >= 0 and i < len(corrected):
                    char = corrected[i]
                    if char.isalpha() and char in dict_char_to_int:
                        corrected[i] = dict_char_to_int[char]
                    elif char in dict_no_change:
                        corrected[i] = dict_no_change[char]
            return ''.join(corrected)

        def preprocess_vintage_series(plate_number):
            if len(plate_number) < 8:
                return plate_number
            corrected = list(plate_number)
            for i in range(2):
                if i < len(corrected):
                    char = corrected[i]
                    if char.isdigit() and char in dict_int_to_char:
                        corrected[i] = dict_int_to_char[char]
                    elif char in dict_no_change:
                        corrected[i] = dict_no_change[char]
            series_end = len(corrected) - 4
            for i in range(4, series_end):
                if i < len(corrected):
                    char = corrected[i]
                    if char.isdigit() and char in dict_int_to_char:
                        corrected[i] = dict_int_to_char[char]
                    elif char in dict_no_change:
                        corrected[i] = dict_no_change[char]
            for i in range(len(corrected) - 4, len(corrected)):
                if i >= 0 and i < len(corrected):
                    char = corrected[i]
                    if char.isalpha() and char in dict_char_to_int:
                        corrected[i] = dict_char_to_int[char]
                    elif char in dict_no_change:
                        corrected[i] = dict_no_change[char]
            return ''.join(corrected)

        def preprocess_bharat_series(plate_number):
            if len(plate_number) < 9:
                return plate_number
            corrected = list(plate_number)
            for i in range(2):
                if i < len(corrected):
                    char = corrected[i]
                    if char.isalpha() and char in dict_char_to_int:
                        corrected[i] = dict_char_to_int[char]
                    elif char in dict_no_change:
                        corrected[i] = dict_no_change[char]
            for i in range(4, 8):
                if i < len(corrected):
                    char = corrected[i]
                    if char.isalpha() and char in dict_char_to_int:
                        corrected[i] = dict_char_to_int[char]
                    elif char in dict_no_change:
                        corrected[i] = dict_no_change[char]
            for i in range(8, len(corrected)):
                char = corrected[i]
                if char.isdigit() and char in dict_int_to_char:
                    corrected[i] = dict_int_to_char[char]
                elif char in dict_no_change:
                    corrected[i] = dict_no_change[char]
            return ''.join(corrected)

        return preprocess_plate_number(prediction)

    def _image_to_base64(self, image_path: str) -> str:
        """Convert image to base64 string"""
        try:
            with open(image_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
                ext = Path(image_path).suffix[1:]
                return f"data:image/{ext};base64,{encoded}"
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to encode image {image_path}: {e}")
            return ""

    def generate_html_report(self, results_data: Dict[str, Any]) -> str:
        """Generate comprehensive HTML report without low confidence section"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        folder_sections = []
        for folder, folder_results in results_data["by_folder"].items():
            folder_rows = self._generate_html_rows(folder_results)
            folder_sections.append(f"""
            <div class="folder-section">
                <h3>📁 {folder}</h3>
                <table>
                    <tr>
                        <th>Original Image</th>
                        <th>Contours</th>
                        <th>Filename</th>
                        <th>Model Prediction</th>
                        <th>EasyOCR Prediction</th>
                    </tr>
                    {folder_rows}
                </table>
            </div>
            """)

        model_info = self._generate_model_info_html()

        total_images = sum(len(folder_results) for folder_results in results_data["by_folder"].values())
        low_conf_count = len(results_data["low_confidence"])

        success_rate = ((total_images - low_conf_count) / total_images * 100) if total_images > 0 else 0.0

        html_content = f"""<!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>License Plate Recognition Report</title>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 30px; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: center; vertical-align: middle; }}
            th {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; font-weight: bold; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            tr:hover {{ background-color: #f0f8ff; }}
            .timestamp {{ color: #666; font-size: 0.9em; margin-bottom: 20px; }}
            .stats {{ display: flex; gap: 20px; margin-bottom: 30px; }}
            .stat-card {{ background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 15px; border-radius: 8px; flex: 1; text-align: center; }}
            .folder-section {{ margin-bottom: 40px; }}
            h1 {{ color: #333; text-align: center; margin-bottom: 30px; }}
            h2 {{ color: #555; }}
            h3 {{ color: #666; margin-top: 30px; }}
            .warning {{ color: #cc0000; }}
            .model-info {{ margin-bottom: 20px; padding: 15px; border-radius: 8px; }}
            .confidence-high {{ background-color: #d4edda; }}
            .confidence-medium {{ background-color: #fff3cd; }}
            .confidence-low {{ background-color: #f8d7da; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚗 License Plate Recognition Report</h1>
            <p class="timestamp">Generated on: {timestamp}</p>

            <div class="stats">
                <div class="stat-card">
                    <strong>Total Images</strong><br>{total_images}
                </div>
                <div class="stat-card">
                    <strong>Low Confidence</strong><br>{low_conf_count}
                </div>
                <div class="stat-card">
                    <strong>Success Rate</strong><br>{success_rate:.2f}%
                </div>
            </div>

            {model_info}

            <h2>📁 Results by Folder</h2>
            {''.join(folder_sections)}
        </div>
    </body>
    </html>"""

        html_path = Path(
            self.config.output_dir) / "reports" / f"plate_recognition_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        self.logger.info(f"✅ HTML report saved: {html_path}")
        return str(html_path)

    def _generate_html_rows(self, results: List[Dict[str, Any]]) -> str:
        """Generate HTML table rows with only EMNIST model prediction as Final Prediction"""
        rows = []
        for result in results:
            conf_class = (
                "confidence-high" if result["model_confidence"] > 0.8
                else "confidence-medium" if result["model_confidence"] > 0.5
                else "confidence-low"
            )
            rows.append(
                f"<tr class='{conf_class}'>"
                f"<td><img src='{result['img_b64']}' width='120' style='border-radius: 4px;'></td>"
                f"<td><img src='{result['contour_img_b64']}' width='120' style='border-radius: 4px;'></td>"
                f"<td>{result['filename']}</td>"
                f"<td><strong>{result.get('model_prediction', '')}</strong><br>"
                f"<span style='font-size:0.9em;color:#888;'>({result.get('model_method', '')}, {result.get('model_confidence', 0.0):.3f})</span></td>"
                f"<td><strong>{result.get('ocr_prediction', '')}</strong><br>"
                f"<span style='font-size:0.9em;color:#888;'>({result.get('ocr_method', '')}, {result.get('ocr_confidence', 0.0):.3f})</span></td>"
                f"</tr>"
            )
        return ''.join(rows)

    def _generate_model_info_html(self) -> str:
        """Generate model information HTML section"""
        if self.model_backend == "keras":
            backend_info = "Keras/TensorFlow"
            backend_style = "background-color: #d4edda; border: 1px solid #c3e6cb;"
        elif self.model_backend == "pytorch":
            backend_info = "PyTorch"
            backend_style = "background-color: #cce5ff; border: 1px solid #b8daff;"
        else:
            backend_info = "Unknown"
            backend_style = "background-color: #f8d7da; border: 1px solid #f5c6cb;"

        if self.model_type == "MNIST":
            return f"""
            <div class="model-info" style="{backend_style}">
                <strong>⚠️ Model: {backend_info} MNIST (Digits Only)</strong>
                <p>This model recognizes digits 0-9 only. EasyOCR is used for full license plate recognition.</p>
            </div>
            """
        elif self.model_type.startswith("EMNIST"):
            return f"""
            <div class="model-info" style="{backend_style}">
                <strong>✅ Model: {backend_info} {self.model_type.replace('_', ' ')} (Letters & Digits)</strong>
                <p>This model recognizes digits 0-9, uppercase A-Z, and lowercase a-z characters (depending on EMNIST subset).</p>
            </div>
            """
        else:
            return f"""
            <div class="model-info" style="{backend_style}">
                <strong>❓ Model: {backend_info} {self.model_type}</strong>
                <p>Model type could not be determined. Results may vary.</p>
            </div>
            """

    def run(self) -> Dict[str, str]:
        """Main execution method"""
        self.logger.info("🚀 Starting optimized license plate recognition")

        results_data = self.process_images_batched()

        report_paths = {}

        if self.config.generate_html:
            report_paths["html"] = self.generate_html_report(results_data)

        total_images = sum(len(folder_results) for folder_results in results_data["by_folder"].values())
        low_conf_count = len(results_data["low_confidence"])

        self.logger.info(f"✅ Processing complete!")
        self.logger.info(f"Total Images: {total_images}, Low Confidence Predictions: {low_conf_count}")
        return report_paths


if __name__ == "__main__":
    # Example Usage:
    # To use a Keras model (e.g., emnist_cnn.h5)
    # config = ProcessingConfig(model_path="emnist_cnn.h5", num_classes=62) # Set num_classes for your model

    # To use a PyTorch model (e.g., emnist_cnn_pytorch.pth)
    config = ProcessingConfig(
        model_path="emnist_torchvision.pth",
        num_classes=36, # IMPORTANT: Set this to 47 for EMNIST_Balanced, 62 for EMNIST_ByClass, etc.
        image_dir="plates_org/test/", # Example directory
        save_contour_images=True # Enable saving contour images
    )

    recognizer = OptimizedPlateRecognizer(config)
    recognizer.run()