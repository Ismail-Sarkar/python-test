import cv2
import numpy as np

# Load pre-trained YOLO model
net = cv2.dnn.readNet("./configs/yolov4.weights", "./configs/yolov4.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Function to calculate volume of a cuboid
def calculate_cuboid_volume(length, width, height):
    return length * width * height

# Process video frames
cap = cv2.VideoCapture("./videos/video.mp4")

# Loop through frames
frame_count = 0
total_volume = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # Object detection
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Calculate bounding box coordinates
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Draw bounding box label
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"{classes[class_id]}: {confidence:.2f}"
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Calculate actual object dimensions
                actual_length = w  # Assuming width represents the length of the object
                actual_width = h   # Assuming height represents the width of the object
                actual_height = (w + h) / 2  # Assuming height is proportional to the average of width and height

                # Calculate volume based on the object's dimensions and the scale of the video
                scale_factor = 1.0  # Example scale factor (adjust as needed based on your calibration or reference objects)
                length_cm = actual_length * scale_factor
                width_cm = actual_width * scale_factor
                height_cm = actual_height * scale_factor

                volume = calculate_cuboid_volume(length_cm, width_cm, height_cm)
                total_volume += volume


    # Display frame
    cv2.imshow("Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Increment frame count
    frame_count += 1

# Calculate average volume
average_volume = total_volume / frame_count

print(f"Average volume of the object: {average_volume} cubic centimeters")

# close video file
cap.release()
cv2.destroyAllWindows()
