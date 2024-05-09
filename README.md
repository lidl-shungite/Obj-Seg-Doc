# Object Segmentation using YOLOv8 and U-Net

This repository contains the code and dataset for segmentation using YOLOv8 and U-Net.

Firstly, we will discuss how to install YOLOv8. Run the following code in your favourite command shell. In my case, I used Anaconda Prompt to install it in the Python virtual environment. 

```shell
pip install ultralytics
```

The next step would be installing the prefered YOLO version. This could be simply done by running the following code.

```py
from ultralytics import YOLO
```
```py
# Create a new YOLO model from scratch
model = YOLO('yolov8n-seg.yaml')

# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8n-seg.pt')
```