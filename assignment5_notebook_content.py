# Assignment 5 (Week 4) - Video Processing and Contour Detection Using OpenCV
# This file contains the content for a Jupyter notebook

# Cell 1 - Title and Introduction
'''
# Assignment 5 (Week 4)

## Video Processing and Contour Detection Using OpenCV

In this assignment, we'll implement three applications using OpenCV:
1. Video I/O Application
2. Motion Detection Application with Foreground Mass and Erosion
3. Contour Detection Application
'''

# Cell 2 - Check and Install Required Packages
'''
# Check and install required packages
import sys
import subprocess
import importlib.util

def check_and_install_package(package_name, import_name=None):
    """Check if a package is installed, and install it if it's not."""
    if import_name is None:
        import_name = package_name
    
    if importlib.util.find_spec(import_name) is None:
        print(f"{package_name} not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"{package_name} installed successfully!")
    else:
        print(f"{package_name} is already installed.")

# List of required packages
required_packages = [
    ("opencv-python", "cv2"),
    ("numpy", "numpy"),
    ("matplotlib", "matplotlib"),
    ("ipython", "IPython")
]

# Check and install each package
for package, import_name in required_packages:
    check_and_install_package(package, import_name)

print("All required packages are installed!")
'''

# Cell 3 - Import Libraries
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from IPython.display import display, HTML

# Verify OpenCV installation
print(f"OpenCV version: {cv2.__version__}")
'''

# Cell 4 - Part 1: Video I/O Application (Introduction)
'''
## Part 1: Video I/O Application

In this part, we'll:
- Read a video file
- Apply transformations to each frame
- Write the processed frames to a new video file

We'll implement a split-screen effect showing the original video on the left and a blurred version on the right.
'''

# Cell 5 - Part 1: Implementation
'''
# Define the input video path
video_path = "../Visuals/Main.mp4"

# Create a VideoCapture object
cap = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
else:
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video Properties:")
    print(f"Width: {frame_width}, Height: {frame_height}")
    print(f"FPS: {fps}")
    print(f"Total Frames: {total_frames}")
    print(f"Duration: {total_frames/fps:.2f} seconds")
    
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_path = "output_part1.avi"
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Process the video
    frame_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Apply transformations:
        # 1. Add a timestamp
        timestamp = f"Frame: {frame_count} | Time: {frame_count/fps:.2f}s"
        cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, (0, 255, 0), 2)
        
        # 2. Add a border
        frame = cv2.rectangle(frame, (0, 0), (frame_width-1, frame_height-1), 
                             (0, 0, 255), 2)
        
        # 3. Apply a slight blur for demonstration
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        
        # 4. Create a split-screen effect: original on left, blurred on right
        combined = frame.copy()
        mid_point = frame_width // 2
        combined[:, mid_point:] = blurred[:, mid_point:]
        
        # Add a dividing line
        cv2.line(combined, (mid_point, 0), (mid_point, frame_height), (0, 255, 255), 2)
        
        # Add labels
        cv2.putText(combined, "Original", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (255, 255, 255), 2)
        cv2.putText(combined, "Blurred", (mid_point + 10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (255, 255, 255), 2)
        
        # Write the frame to the output video
        out.write(combined)
        
        # Display progress
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")
        
        frame_count += 1
    
    # Calculate processing time
    elapsed_time = time.time() - start_time
    print(f"Processing completed in {elapsed_time:.2f} seconds")
    print(f"Average processing speed: {frame_count/elapsed_time:.2f} frames per second")
    
    # Release resources
    cap.release()
    out.release()
    
    print(f"Output video saved to: {output_path}")
    
    # Display a sample frame (we'll use the last processed frame)
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    plt.title("Sample Processed Frame")
    plt.axis('off')
    plt.show()
'''

# Cell 6 - Part 2: Motion Detection Application (Introduction)
'''
## Part 2: Motion Detection Application with Foreground Mass and Erosion

In this part, we'll:
- Create a background model from initial frames
- Compute the foreground mass (difference between current frame and background)
- Apply erosion to remove noise
- Annotate frames with bounding boxes around motion areas

### Explanation of Concepts:

**Foreground Mass**: This is the difference between the current frame and the background model. It helps identify moving objects in the scene.

**Erosion**: A morphological operation that removes small, isolated pixels (noise) from the foreground mask. It works by shrinking the boundaries of foreground objects, effectively eliminating small noise pixels while preserving larger motion areas.
'''

# Cell 7 - Part 2: Implementation
'''
# Define the input video path
video_path = "../Visuals/Main.mp4"

# Create a VideoCapture object
cap = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
else:
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_path = "output_part2.avi"
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height*2))
    
    # Initialize variables for background subtraction
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
    else:
        # Convert to grayscale for background modeling
        prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Create a background model by averaging the first few frames
        num_bg_frames = 10
        bg_model = np.zeros_like(prev_gray, dtype=np.float32)
        
        for i in range(num_bg_frames):
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bg_model += gray / num_bg_frames
        
        # Convert background model to uint8
        bg_model = bg_model.astype(np.uint8)
        
        # Reset video to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Process the video
        frame_count = 0
        
        # Create a kernel for erosion
        kernel = np.ones((5, 5), np.uint8)
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Convert current frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Compute foreground mask (difference between current frame and background)
            foreground_mask = cv2.absdiff(gray, bg_model)
            
            # Apply threshold to get binary mask
            _, thresh = cv2.threshold(foreground_mask, 30, 255, cv2.THRESH_BINARY)
            
            # Apply erosion to remove noise
            eroded_mask = cv2.erode(thresh, kernel, iterations=1)
            
            # Find contours in the eroded mask
            contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Create a copy of the original frame for drawing
            result_frame = frame.copy()
            
            # Draw bounding boxes around detected motion areas
            for contour in contours:
                # Filter out small contours
                if cv2.contourArea(contour) > 100:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(result_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Add text to explain what's happening
            cv2.putText(result_frame, "Motion Detection", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Create visualization of masks
            # Convert masks to 3-channel for display
            foreground_mask_color = cv2.cvtColor(foreground_mask, cv2.COLOR_GRAY2BGR)
            eroded_mask_color = cv2.cvtColor(eroded_mask, cv2.COLOR_GRAY2BGR)
            
            # Add labels to the masks
            cv2.putText(foreground_mask_color, "Foreground Mask", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(eroded_mask_color, "Eroded Mask", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Create a split view: top half shows original with bounding boxes, 
            # bottom half shows masks side by side
            masks_combined = np.hstack((foreground_mask_color, eroded_mask_color))
            
            # Resize if necessary to match frame width
            if masks_combined.shape[1] != frame_width:
                masks_combined = cv2.resize(masks_combined, (frame_width, frame_height))
            
            # Stack the result frame and masks vertically
            output_frame = np.vstack((result_frame, masks_combined))
            
            # Write the frame to the output video
            out.write(output_frame)
            
            # Update frame count
            frame_count += 1
            
            # Display progress
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames")
        
        # Release resources
        cap.release()
        out.release()
        
        print(f"Output video saved to: {output_path}")
        
        # Display the last output frame
        plt.figure(figsize=(12, 16))
        plt.imshow(cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB))
        plt.title("Motion Detection with Foreground Mass and Erosion")
        plt.axis('off')
        plt.show()
'''

# Cell 8 - Part 3: Contour Detection Application (Introduction)
'''
## Part 3: Contour Detection Application

In this part, we'll:
- Extract a frame from the video to use as our image
- Convert the image to grayscale and threshold it
- Detect contours using cv2.findContours
- Draw the contours on the original image
- (Bonus) Extend the solution to detect and draw contours in a video

### Explanation of Contours:

Contours are curves joining all continuous points along a boundary that have the same color or intensity. They are useful for shape analysis, object detection, and recognition.
'''

# Cell 9 - Part 3: Implementation
'''
# Part 3A: Contour detection on a single frame

# Extract a frame from the video to use as our image
video_path = "../Visuals/Main.mp4"
cap = cv2.VideoCapture(video_path)

# Skip to a frame with interesting content
cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
ret, frame = cap.read()

if not ret:
    print("Error: Could not read frame from video.")
else:
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply binary threshold
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Found {len(contours)} contours")
    
    # Create a copy of the original frame for drawing
    contour_image = frame.copy()
    
    # Draw all contours
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    
    # Add text explaining contours
    cv2.putText(contour_image, f"Contours: {len(contours)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Create a side-by-side comparison
    comparison = np.hstack((frame, contour_image))
    
    # Display the comparison
    plt.figure(figsize=(16, 8))
    plt.imshow(cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB))
    plt.title("Original Image vs. Contour Detection")
    plt.axis('off')
    plt.show()
    
    # Explanation of the contour detection process
    print("Contour Detection Process:")
    print("1. Convert the image to grayscale to simplify processing")
    print("2. Apply Gaussian blur to reduce noise and improve contour detection")
    print("3. Apply binary threshold to create a black and white image")
    print("4. Find contours using cv2.findContours")
    print("5. Draw the contours on the original image")
    
    # Explanation of the contour hierarchy
    print("\nContour Hierarchy:")
    print("The hierarchy describes the relationship between contours.")
    print("For each contour, it stores information about:")
    print("- Next contour at the same level")
    print("- Previous contour at the same level")
    print("- First child contour")
    print("- Parent contour")
'''

# Cell 10 - Part 3: Bonus - Contour Detection on Video
'''
# Part 3B: Bonus - Contour detection on video

# Reset video to beginning
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_path = "output_part3.avi"
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width*2, frame_height))

# Process the video
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply threshold
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a copy for drawing
    contour_frame = frame.copy()
    
    # Draw contours
    cv2.drawContours(contour_frame, contours, -1, (0, 255, 0), 2)
    
    # Add text
    cv2.putText(contour_frame, f"Contours: {len(contours)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Create side-by-side view
    output_frame = np.hstack((frame, contour_frame))
    
    # Write to video
    out.write(output_frame)
    
    # Update frame count
    frame_count += 1
    
    # Display progress
    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames")

# Release resources
cap.release()
out.release()

print(f"Output video saved to: {output_path}")
'''

# Cell 11 - Conclusion
'''
## Conclusion

In this assignment, we implemented three OpenCV applications:

1. **Video I/O Application**: We read a video file, applied transformations (timestamp, border, split-screen with blur), and wrote the processed frames to a new video file.

2. **Motion Detection Application**: We created a background model, computed the foreground mass, applied erosion to remove noise, and annotated frames with bounding boxes around motion areas.

3. **Contour Detection Application**: We detected and drew contours on both a single frame and throughout a video.

These applications demonstrate the power and versatility of OpenCV for video processing and computer vision tasks.
''' 