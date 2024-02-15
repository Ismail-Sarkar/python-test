import cv2
import numpy as np

net = cv2.dnn.readNet("./configs/yolov4.weights", "./configs/yolov4.cfg")
classes = []
with open("coco.names","r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

def calculate_cuboid_volume(length, width, height):
    return length * width * height


object_length = 0.1
object_width = 0.05
object_height = 0.15

cap = cv2.VideoCapture("./videos/video1.mp4")
frame_count = 0
total_volume = 0


while True:
    ret, frame = cap.read()
    if not ret:
        break
    height, width, _=frame.shape


    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)


    max_confidence = 0
    max_detection = None

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and confidence > max_confidence:
                max_confidence = confidence
                max_detection = detection

                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Assign class label
                label = f"{classes[class_id]}: {confidence:.2f}"
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                scale_factor = 1.0
                length_pixels = w * scale_factor
                width_pixels = h * scale_factor
                height_pixels = (w + h) / 2 * scale_factor
                length_centimeters = length_pixels * 100 * object_length / object_width
                width_centimeters = width_pixels * 100 * object_width / object_width
                height_centimeters = height_pixels * 100 * object_height / object_width
                volume = calculate_cuboid_volume(length_centimeters, width_centimeters, height_centimeters)
                total_volume += volume
                frame_count += 1

    cv2.imshow("Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

average_volume = total_volume / frame_count if frame_count > 0 else 0
print(f"Average volume of the object: {average_volume} cubic centimeters")

cap.release()
cv2.destroyAllWindows()

