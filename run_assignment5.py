import sys
import os

# Ensure the current directory is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the solution functions
from assignment5_solution import part1_video_io, part2_motion_detection, part3_contour_detection, main

if __name__ == "__main__":
    print("Running Assignment 5 Solution")
    main() 