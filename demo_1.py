import cv2
import numpy as np


# Load pre-trained YOLO model
net = cv2.dnn.readNet("./configs/yolov4.weights", "./configs/yolov4.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


# Load image
image = cv2.imread("./images/123.jpg")

# Object detection
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop = False)
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
            center_x = int(detection[0] * image.shape[1])
            center_y = int(detection[1] * image.shape[0])
            w = int(detection[2] * image.shape[1])
            h = int(detection[3] * image.shape[0])

            # Calculate dimensions (assuming object is oriented horizontally)
            length = w
            width = h

            # Display bounding box and dimensions
            cv2.rectangle(image, (center_x - w//2, center_y - h//2), (center_x + w//2, center_y + h//2), (0, 255, 0), 2)
            cv2.putText(image, f"Length: {length} pixels, Width: {width} pixels", (center_x - w//2, center_y - h//2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


# Display image with bounding box and dimensions
cv2.imshow("Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
