# Object Segmentation using YOLOv8 and U-Net

This repository contains the code and dataset for segmentation using YOLOv8 and U-Net.

## YOLOv8

Firstly, we will discuss how to install YOLOv8. Run the following code in your favourite command shell. In my case, I used Anaconda Prompt to install it in the Python virtual environment. 

```shell
pip install ultralytics
```

The next step would be installing the prefered YOLO version. This could be simply done by running the following code.

```py
from ultralytics import YOLO
```

In the following code, 'yolov8n-seg.yaml' and 'yolov8n-seg.pt' are not only model version and mode, but also a path. If the file doesn't exist, the file will be downloaded and placed. 

```py
# Create a new YOLO model from scratch
model = YOLO('yolov8n-seg.yaml')

# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8n-seg.pt')
```

Before you start training your model, you must first build a 'config.yaml' file which contains the paths to the train and validation data, and a list of all classes in the dataset. The parameters 'epochs' determine for how much epoch would your model train and 'imgsz' to adjust the size of all training images to be the same.

```py
#Set and adjust epoch and imgsz as to your need
results = model.train(data='config.yaml', epochs=100, imgsz=640)
```

If you are interested in how training and testing escalated, I recommend checking this [documentation](https://obj-seg-doc-e3wipu72g6lsyt3rvkxp2g.streamlit.app/).

Libraries that are required in order to run the code, are listed down in "requirements.txt" file. Install these libraries or packages in the same manner as you installed ultralytics. Run ".py" file. Quick reminder that the program may vary according to the specifications of your device that you are running the program on. 

## U-Net

