import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time

# Load the trained CNN model
CNN_model = keras.models.load_model('CNN_model.h5')

def get_img_contour_thresh(img):
    # Define a fixed size for processing
    x, y, w, h = 100, 100, 400, 400
    
    # Get the ROI first
    roi_color = img[y:y + h, x:x + w]
    
    # Convert to grayscale
    gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Apply Otsu's thresholding
    _, thresh1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Apply morphological operations to clean up the image
    kernel = np.ones((5,5), np.uint8)
    thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
    thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return img, contours, thresh1, gray

def is_digit_like(contour, min_area=2000, max_area=40000):
    # Get contour area
    area = cv2.contourArea(contour)
    
    # Check if area is within reasonable range for a digit
    if area < min_area or area > max_area:
        return False
    
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)
    
    # Check aspect ratio (height/width should be between 1.0 and 2.5 for digits)
    aspect_ratio = h / w
    if aspect_ratio < 1.0 or aspect_ratio > 2.5:
        return False
    
    # Check solidity (area / convex hull area)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0
    if solidity < 0.6:  # Digits usually have high solidity
        return False
    
    # Check extent (area / bounding rectangle area)
    extent = float(area) / (w * h)
    if extent < 0.4:  # Digits usually fill their bounding box reasonably well
        return False
    
    return True

def draw_prediction_box(img, prediction, confidence, x, y, w, h, all_probabilities):
    # Create a semi-transparent overlay for the prediction box
    overlay = img.copy()
    
    # Draw a filled rectangle for the prediction box
    cv2.rectangle(overlay, (x+w+10, y), (x+w+300, y+200), (0, 0, 0), -1)
    
    # Add the overlay with transparency
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
    
    # Draw the prediction and confidence
    cv2.putText(img, "Detected:", (x+w+20, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(img, str(prediction), (x+w+70, y+90), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 3)
    cv2.putText(img, f"Confidence: {confidence:.1f}%", (x+w+20, y+130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw top 3 predictions
    sorted_indices = np.argsort(all_probabilities)[-3:][::-1]
    y_offset = 160
    for idx in sorted_indices:
        prob = all_probabilities[idx] * 100
        cv2.putText(img, f"#{idx}: {prob:.1f}%", (x+w+20, y+y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 25

def process_digit_roi(roi, target_size=28):
    if roi is None or roi.size == 0:
        return None
        
    # Add padding to make the digit more centered
    border_size = int(roi.shape[0] * 0.2)  # 20% padding
    padded = cv2.copyMakeBorder(roi, border_size, border_size, border_size, border_size,
                               cv2.BORDER_CONSTANT, value=0)
    
    # Resize to target size (28x28 for MNIST)
    resized = cv2.resize(padded, (target_size, target_size))
    
    # Ensure the digit is white on black background (MNIST format)
    if cv2.countNonZero(resized) > resized.size / 2:
        resized = cv2.bitwise_not(resized)
    
    # Enhance contrast
    resized = cv2.normalize(resized, None, 0, 255, cv2.NORM_MINMAX)
    
    return resized

def main():
    # Try different camera indices
    camera_index = 0
    max_attempts = 3
    cap = None
    
    while camera_index < max_attempts:
        print(f"Trying camera index {camera_index}...")
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"Failed to open camera {camera_index}")
            camera_index += 1
            continue
            
        # Try to read a test frame
        ret, test_frame = cap.read()
        if ret and test_frame is not None and test_frame.size > 0:
            print(f"Successfully opened camera {camera_index}")
            break
            
        print(f"Camera {camera_index} opened but couldn't read frame")
        cap.release()
        camera_index += 1
    
    if cap is None or not cap.isOpened():
        print("Error: Could not open any camera")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Wait for camera to initialize
    time.sleep(2)
    
    print("Camera initialized. Starting capture...")
    
    # Variables for prediction stability
    last_prediction = None
    prediction_count = 0
    stable_threshold = 3  # Reduced threshold for faster response
    
    # Create windows
    cv2.namedWindow("Digit Recognition", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Digit Recognition", 1280, 720)
    cv2.namedWindow("Debug View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Debug View", 800, 600)
    
    while True:
        ret, img = cap.read()
        if not ret or img is None:
            print("Failed to grab frame")
            break
            
        if img.size == 0:
            print("Empty frame received")
            continue
        
        # Define the rectangle coordinates
        x, y = 100, 100  # Starting point
        w, h = 400, 400  # Width and height
        
        try:
            # Process the image
            img, contours, thresh, gray = get_img_contour_thresh(img)
            
            # Initialize prediction
            prediction = None
            confidence = 0
            all_probabilities = None
            processed_digit = None
            
            # Process contours if any are found
            if len(contours) > 0:
                # Filter contours by digit-like characteristics
                valid_contours = [cnt for cnt in contours if is_digit_like(cnt)]
                
                if valid_contours:
                    # Sort contours by area and get the largest
                    contour = max(valid_contours, key=cv2.contourArea)
                    
                    # Get bounding rectangle
                    x_cont, y_cont, w_cont, h_cont = cv2.boundingRect(contour)
                    
                    # Extract the digit region
                    digit_roi = thresh[y_cont:y_cont + h_cont, x_cont:x_cont + w_cont]
                    
                    if digit_roi.size > 0:
                        # Process the ROI
                        processed_digit = process_digit_roi(digit_roi)
                        
                        if processed_digit is not None:
                            # Normalize and reshape for prediction
                            input_data = processed_digit.astype('float32') / 255.0
                            input_data = input_data.reshape(1, 28, 28, 1)
                            
                            # Make prediction
                            pred_probabilities = CNN_model.predict(input_data, verbose=0)
                            all_probabilities = pred_probabilities[0]
                            prediction = np.argmax(all_probabilities)
                            confidence = all_probabilities[prediction] * 100
                            
                            # Check prediction stability with higher confidence threshold
                            if prediction == last_prediction and confidence > 85:  # Increased confidence threshold
                                prediction_count += 1
                            else:
                                prediction_count = 0
                                last_prediction = prediction
                            
                            # Draw contour for debugging
                            cv2.drawContours(img[y:y+h, x:x+w], [contour], -1, (0, 255, 0), 2)
            
            # Draw the rectangle
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            
            # Add text for instructions
            cv2.putText(img, "Write a single digit clearly in the box", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show prediction if stable
            if prediction is not None and prediction_count >= stable_threshold:
                draw_prediction_box(img, prediction, confidence, x, y, w, h, all_probabilities)
            
            # Create debug view
            debug_view = np.zeros((600, 800), dtype=np.uint8)
            
            # Show threshold image
            if thresh is not None and thresh.size > 0:
                resized_thresh = cv2.resize(thresh, (400, 400))
                debug_view[0:400, 0:400] = resized_thresh
            
            # Show processed digit
            if processed_digit is not None:
                large_digit = cv2.resize(processed_digit, (200, 200))
                debug_view[400:600, 0:200] = large_digit
            
            # Show the frames
            cv2.imshow("Digit Recognition", img)
            cv2.imshow("Debug View", debug_view)
            
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            continue
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 