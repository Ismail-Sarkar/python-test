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

object_length = 0.1  # Example length
object_width  = 0.05 # Example width
object_height = 0.15 # Example height

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

    # Display frame
    cv2.imshow("Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Calculate volume based on the object's dimensions and the scale of the video
    scale_factor  = 1.0 # Example scale factor (adjust as needed based on your calibration or reference objects)
    length_pixels = w * scale_factor
    width_pixels  = h * scale_factor
    height_pixels = (w + h) / 2 * scale_factor # Assuming height is proportional to the average of width and height
    # length_meters = length_pixels * object_length / object_width # Convert pixels to meters using known dimensions
    length_centimeters = length_pixels * 100 * object_length / object_width # Convert pixels to centimeters and meters to centimeters
    # width_meters  = width_pixels * object_width / object_width   # Convert pixels to meters using known dimensions
    width_centimeters  = width_pixels * 100 * object_width / object_width     # Convert pixels to centimeters and meters to centimeters
    # height_meters = height_pixels * object_height / object_width # Convert pixels to meters using known dimensions
    height_centimeters = height_pixels * 100 * object_height / object_width # Convert pixels to centimeters and meters to centimeters
    # volume = calculate_cuboid_volume(length_meters, width_meters, height_meters)
    volume = calculate_cuboid_volume(length_centimeters, width_centimeters, height_centimeters)
    # total_volume += volume
    total_volume += volume

    # Increment frame count
    frame_count += 1

# close video file
# cap.release()
# cv2.destroyAllWindows()

# Calculate average volume
average_volume = total_volume / frame_count

# print(f"Average volume of the object: {average_volume} cubic meters")
print(f"Average volume of the object: {average_volume} cubic centimeters")



# close video file
cap.release()
cv2.destroyAllWindows()