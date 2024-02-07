import cv2
import numpy as np

# Load YOLOv3 model
model_weights = './configs/yolov4.weights'  # Path to YOLOv3 pre-trained weights
model_config = './configs/yolov4.cfg'       # Path to YOLOv3 configuration file
net = cv2.dnn.readNet(model_weights, model_config)

# Load image
image_path = './images/123.jpg'          # Path to your input image
image = cv2.imread(image_path)
print(image)

# Preprocess image
blob = cv2.dnn.blobFromImage(image, 1.0/255.0, (416, 416), swapRB=True, crop=False)
# print(blob)
net.setInput(blob)

# Get detection results
output_layers = net.getUnconnectedOutLayersNames()
outputs = net.forward(output_layers)

# Interpret detection results and draw bounding boxes
conf_threshold = 0.1  # Confidence threshold for detected objects
nms_threshold = 1    # Non-maximum suppression threshold
class_ids = []
confidences = []
boxes = []


for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        
        if confidence > conf_threshold:
            center_x = int(detection[0] * image.shape[1])
            center_y = int(detection[1] * image.shape[0])
            width = int(detection[2] * image.shape[1])
            height = int(detection[3] * image.shape[0])
            
            x = int(center_x - width / 2)
            y = int(center_y - height / 2)
            
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, width, height])

# Apply non-maximum suppression
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

# Draw bounding boxes and labels
colors = np.random.uniform(0, 255, size=(len(class_ids), 3))

classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 
           'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
            'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 
            'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 
            'teddy bear', 'hair drier', 'toothbrush','refrigerator', 'microwave', 'oven', 
            'coffee maker', 'blender', 'washing machine', 'vacuum cleaner', 'iron', 
            'television', 'lamp', 'fan', 'air purifier', 'kettle', 'hair dryer','phone','frame',
            'photo'
]

for i in indices:
    # i = i[0]
    box = boxes[i]
    x, y, w, h = box
    label = str(classes[class_ids[i]])
    print(label)
    color = colors[i]
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Display the result
cv2.imshow('Object Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
