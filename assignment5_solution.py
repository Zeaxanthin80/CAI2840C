import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# Part 1: Video I/O Application
def part1_video_io():
    # Define the input video path
    video_path = "Visuals/Main.mp4"
    
    # Create a VideoCapture object
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
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
    
    # Return the last frame for display
    return combined

# Part 2: Motion Detection Application with Foreground Mass and Erosion
def part2_motion_detection():
    # Define the input video path
    video_path = "Visuals/Main.mp4"
    
    # Create a VideoCapture object
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
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
        return
    
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
    
    # Return the last output frame for display
    return output_frame

# Part 3: Contour Detection Application
def part3_contour_detection():
    # Part 3A: Contour detection on a single frame
    
    # Extract a frame from the video to use as our image
    video_path = "Visuals/Main.mp4"
    cap = cv2.VideoCapture(video_path)
    
    # Skip to a frame with interesting content
    cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame from video.")
        return
    
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
    
    # Return the comparison image for display
    return comparison

# Main function to run all parts
def main():
    print("Running Part 1: Video I/O Application")
    part1_result = part1_video_io()
    
    print("\nRunning Part 2: Motion Detection Application")
    part2_result = part2_motion_detection()
    
    print("\nRunning Part 3: Contour Detection Application")
    part3_result = part3_contour_detection()
    
    # Display results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 1, 1)
    plt.imshow(cv2.cvtColor(part1_result, cv2.COLOR_BGR2RGB))
    plt.title("Part 1: Video I/O Application")
    plt.axis('off')
    
    plt.subplot(3, 1, 2)
    plt.imshow(cv2.cvtColor(part2_result, cv2.COLOR_BGR2RGB))
    plt.title("Part 2: Motion Detection Application")
    plt.axis('off')
    
    plt.subplot(3, 1, 3)
    plt.imshow(cv2.cvtColor(part3_result, cv2.COLOR_BGR2RGB))
    plt.title("Part 3: Contour Detection Application")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("assignment5_results.png")
    plt.show()

if __name__ == "__main__":
    main() 