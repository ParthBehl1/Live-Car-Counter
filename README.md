This project implements a car counting system using a pre-trained YOLO (You Only Look Once) object detection model. 
It can process live video streams or uploaded videos to count the number of cars present in each frame.

Key Features:

Real-time or Pre-recorded Video Processing: Count cars in live video streams from a webcam or analyze pre-recorded video files.
YOLO Object Detection: Leverage a pre-trained YOLO model (e.g., YOLOv5) for efficient car detection.
Car Counting: Track and display the total number of cars identified in each frame.
Customizable Masking (Optional): Mask the area (using bitwise_and) you want to focus on detecting from the real time video or recorded video.[refer cars.mp4 then mask.png]

Requirements:
Python 3.8 (as latest versions does not have mediapipe support)
OpenCV-Python (pip install opencv-python)
NumPy (pip install numpy)
YOLO model weights (Download a pre-trained car detection YOLO model, e.g., YOLOv8n)
cvzone 
