MODEL_TYPE = "cv2"
weightsPath= "yolo-coco/yolov3.weights"
configPath = "yolo-coco/yolov3.cfg"
inputVideoPath = "test_videos/bridge.mp4"
outputVideoPath = "outputVideos/nyc.mp4"
preDefinedConfidence = 0.5
preDefinedThreshold = 0.5
RUN_EVERY = 1
show_video = True
FRAMES_BEFORE_CURRENT = 5
inputWidth, inputHeight = 416, 416
horizontal_margin = 8
vertical_margin = 15
distance_margin = 20
BATCH_SIZE = 2
LABELS = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
          'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
          'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
          'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
          'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
          'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa',
          'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard',
          'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
          'teddy bear', 'hair drier', 'toothbrush']
objects_of_interest = ["bicycle", "car", "motorbike", "bus", "truck", "train"]
