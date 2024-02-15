from Detector import *
import os

def main():
    videoPath = "./videos/video.mp4"

    configPath = os.path.join("configs", "yolov4.cfg")
    modelPath = os.path.join("configs", "yolov4.weights")
    classesPath = os.path.join("coco.names")

    Detector(videoPath, configPath, modelPath, classesPath)

if __name__ == '__main__':
    main()

