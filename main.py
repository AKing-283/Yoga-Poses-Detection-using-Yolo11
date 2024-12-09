import cv2
import cvzone
import math
import os
from ultralytics import YOLO

# Path to your model (update with the correct model path)
model_path = "/Users/pushpakreddy/Downloads/train27/weights/best.pt"  # Replace with the actual path to your YOLO model
model = YOLO(model_path)

# List of yoga poses you want to detect (16 classes)
classnames = [
    'chair_pose', 'dolphin_plank_pose', 'downward-facing_dog_pose', 'fish_pose', 'goddess_pose',
    'locust_pose', 'lord_of_the_dance_pose', 'low_lunge_pose', 'seated_forward_bend_pose', 'side_plank_pose',
    'staff_pose', 'tree_pose', 'warrior_1_pose', 'warrior_2_pose', 'warrior_3_pose', 'wide_angle_seated_forward_bend_pose'
]

def process_frame(frame):
    """Function to process each frame and detect yoga poses."""
    results = model(frame)
    for info in results:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Get the confidence of the prediction
            confidence = box.conf[0]
            conf = math.ceil(confidence * 100)

            # Get the detected class
            class_detect = box.cls[0]
            class_detect = int(class_detect)
            class_detect = classnames[class_detect]

            # Display the result if confidence > 40%
            if conf > 40:
                # Draw bounding box around the detected pose
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Display the pose name and confidence on the video frame
                cvzone.putTextRect(frame, f'{class_detect} {conf}%', [x1 + 8, y1 - 12], thickness=2, scale=1)

    return frame

def detect_from_video(video_path):
    """Process a video file for yoga pose detection."""
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()

        # Replay video when it ends
        if not ret:
            cap = cv2.VideoCapture(video_path)  # Restart the video
            continue

        # Resize the frame
        frame = cv2.resize(frame, (640, 480))

        # Process the frame
        frame = process_frame(frame)

        # Show the frame with detection
        cv2.imshow('Yoga Pose Detection (Video)', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def detect_from_image(image_path):
    """Process an image for yoga pose detection."""
    image = cv2.imread(image_path)
    image = cv2.resize(image, (640, 480))  # Resize the image to a fixed size

    # Process the image
    image = process_frame(image)

    # Show the image with detection
    cv2.imshow('Yoga Pose Detection (Image)', image)
    cv2.waitKey(0)  # Wait until a key is pressed
    cv2.destroyAllWindows()

# Main code to detect from either video or image
input_path = "/Users/pushpakreddy/Downloads/311742532-52484a06-559f-4d4d-a4b3-eaa949782729.jpeg"  # Set the path to either a video or an image

if os.path.isfile(input_path):  # Check if the input is a file (either video or image)
    file_extension = input_path.split('.')[-1].lower()
    
    if file_extension in ['mp4', 'avi', 'mov']:  # Check if the input is a video
        detect_from_video(input_path)
    elif file_extension in ['jpg', 'jpeg', 'png']:  # Check if the input is an image
        detect_from_image(input_path)
    else:
        print("Unsupported file type.")
else:
    print(f"File {input_path} not found.")
