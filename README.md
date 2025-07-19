# License Plate Recognition System

This project provides tools for recognizing license plate characters and generating HTML reports of the results.

## Overview

The system consists of the following components:

1. **MNIST CNN Model**: A convolutional neural network trained on the MNIST dataset for digit recognition (0-9).
2. **EasyOCR Integration**: A deep learning-based OCR system used for alphabet recognition and as a fallback method.
3. **Batch Plate Reader**: A script that processes multiple license plate images, recognizes characters using a hybrid approach, and generates an HTML report.
4. **License Plate Reader**: A script for processing individual license plate images.

## Setup and Installation

### Prerequisites

- Python 3.9+
- TensorFlow 2.x
- OpenCV (cv2)
- NumPy
- Matplotlib (for visualization)

### Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install tensorflow opencv-python numpy matplotlib easyocr
   ```
   
   Note: EasyOCR is used for alphabet recognition and as a fallback method. It may require additional dependencies depending on your system.

## Usage

### Training the MNIST Model

The system uses a CNN model trained on the MNIST dataset for digit recognition. To train or retrain the model:

```
python train_emnist_model.py
```

This script will:
1. Download and prepare the MNIST dataset
2. Train a CNN model for digit recognition
3. Save the trained model as `emnist_cnn.h5`
4. Generate a training history plot as `training_history.png`

The model achieves approximately 99% accuracy on the MNIST test set.

### Processing License Plates in Batch

To process multiple license plate images and generate an HTML report:

```
python batch_plate_reader_html.py
```

This script will:
1. Load the trained MNIST model
2. Process all images in the `plates/` directory (including subdirectories)
3. Generate an HTML report (`plate_predictions.html`) with:
   - Low confidence predictions highlighted
   - Results grouped by folder
   - Images, predictions, confidence scores, and methods used

### Processing Individual License Plates

To process a single license plate image:

```
python license_plate_reader.py
```

Edit the script to specify the image path in the `IMG_PATH` variable.

## How It Works

1. **Character Segmentation**: The system uses OpenCV to:
   - Convert the image to grayscale
   - Apply Gaussian blur
   - Apply thresholding
   - Find contours
   - Extract individual character regions

2. **Character Recognition**: The system uses a hybrid approach:
   - **Primary Method (MNIST)**: Segmented characters are:
     - Resized to 28x28 pixels
     - Normalized
     - Passed to the MNIST CNN model for prediction
   - **Secondary Method (EasyOCR)**: Used in the following cases:
     - When the MNIST model predicts potential letters (indices >= 10)
     - When character segmentation fails
     - When the MNIST model is not available
     - When higher confidence is achieved with EasyOCR

3. **Report Generation**: The results are:
   - Compiled into an HTML report
   - Sorted by confidence score
   - Grouped by folder
   - Method used (MNIST or EasyOCR) is indicated for each prediction

## Troubleshooting

### Common Issues

1. **"No segments found"**: This occurs when the character segmentation algorithm cannot identify individual characters in the image. Possible solutions:
   - Improve image quality
   - Adjust the contour filtering parameters in the `segment_characters` function

2. **"Model not loaded"**: This occurs when the MNIST model file (`emnist_cnn.h5`) is missing or corrupted. Solutions:
   - Run `train_emnist_model.py` to generate a new model
   - Check file permissions

## Limitations

- The MNIST model is trained only on digits (0-9), but we use EasyOCR as a fallback for letters
- Character segmentation may fail on complex backgrounds or poor quality images
- EasyOCR processing is slower than the MNIST model, especially without GPU acceleration
- The system works best with clear, well-lit images of license plates

## Future Improvements

- Train a custom model on the full EMNIST dataset to improve letter recognition without relying on EasyOCR
- Implement more robust character segmentation algorithms
- Add support for different license plate formats and layouts
- Implement a web interface for easier use
- Add confidence thresholds to automatically choose between MNIST and EasyOCR based on prediction confidence

## License

This project is licensed under the MIT License - see the LICENSE file for details.