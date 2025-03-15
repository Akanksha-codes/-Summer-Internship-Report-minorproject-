# Digit Recognition with CNN

A real-time digit recognition system using Convolutional Neural Networks (CNN) and OpenCV. The system captures input from a webcam and recognizes handwritten digits with high accuracy.

## Features

- Real-time digit recognition using webcam
- Debug view for visualization of image processing steps
- Top 3 predictions with confidence scores
- Robust digit detection with noise reduction
- MNIST format compatibility

## Requirements

- Python 3.9+
- OpenCV
- TensorFlow
- NumPy
- Keras

## Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
cd <repository-name>
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install opencv-python tensorflow numpy
```

4. Make sure you have the trained model file `CNN_model.h5` in the project directory.

## Usage

1. Run the script:
```bash
python test_cnn.py
```

2. Write a digit (0-9) on white paper with a dark marker
3. Hold it up to the webcam within the green rectangle
4. The system will display the recognized digit with confidence scores
5. Press 'q' to quit the application

## Debug View

The application includes a debug window that shows:
- Processed image (top half)
- MNIST format digit (bottom left)
- Top 3 predictions with confidence scores

## Camera Access

On macOS, you may need to grant camera access permissions:
1. Go to System Preferences > Security & Privacy > Privacy > Camera
2. Enable camera access for Terminal or your Python IDE 