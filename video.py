import cv2
import numpy as np

# Load YOLOv3 model
model_weights = './configs/yolov4.weights'  # Path to YOLOv3 pre-trained weights
model_config = './configs/yolov4.cfg'       # Path to YOLOv3 configuration file
net = cv2.dnn.readNet(model_weights, model_config)

# Open video file
video_path = './videos/video1.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
 # Preprocess frame
    blob = cv2.dnn.blobFromImage(frame, 1.0 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

 # Get detection results
    output_layers = net.getUnconnectedOutLayersNames()
    outputs = net.forward(output_layers)

    # Interpret detection results and draw bounding boxes
    conf_threshold = 0  # Confidence threshold for detected objects
    nms_threshold = 0    # Non-maximum suppression threshold
    class_ids = []
    confidences = []
    boxes = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > conf_threshold:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])
                
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, width, height])



    # Apply non-maximum suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    print(indices)

    # Draw bounding boxes and labels
    colors = np.random.uniform(0, 255, size=(len(class_ids), 3))

    classes = [
    'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'sofa', 'potted plant', 'bed', 'dining table', 'toilet', 'tvmonitor', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]



    # Draw bounding boxes and labels
    for i in indices:
        # i = i[0]
        box = boxes[i]
        x, y, w, h = box
        label = str(classes[class_ids[i]])
        # label = str(class_ids[i])
        print(label)
        color = colors[i]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display frame with detections
    cv2.imshow('Object Detection', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()


