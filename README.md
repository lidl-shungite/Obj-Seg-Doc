# Object Segmentation using YOLOv8 and U-Net

This repository contains the code and dataset for segmentation using YOLOv8 and U-Net. The PASCAL dataset is taken from [kaggle](https://www.kaggle.com/datasets/gopalbhattrai/pascal-voc-2012-dataset)

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

Libraries that are required in order to run the code, are listed down in "requirements.txt" file. Install these libraries or packages in the same manner as you installed ultralytics. After that there is a file named ".py", it is a python program built to segment objects out of background using OpenCV, please have it in mind that segmentation may not be accurate. Run ".py" file. A quick reminder that the program may vary according to the specifications of your device that you are running the program on. 

## U-Net

The name "U-Net" surfaced on the internet in the year 2015, its original purpose was to segment out brain tumors of an x-ray image. People quickly realized that it could be used for other purposes, namely object segmentation and other forms of segmentation. In the ".ipynb" file lies a bit modified U-Net architecture or my version of the architecture. 

### Data Preprocessing for U-Net

If you have checked out the dataset, by now you would know that the masks are in black and some other colors, you don't want that. When training this would make the training process more tedious and long, making it more complex for the model. Of course, this process is not necessary if you are trying to classify the classes of the segmented object as well, as opposed to this, I wanted only segmentation, so I had to change the masks to black and white. Computers don't unterstand colors like we do, it uses the RGB model which has 8-bits (2<sup>8</sup> or 256 color values ranging from 0 to 255) for three color channels; in total 256<sup>3</sup> or 16,777,216 possible colors. The mask is a binary image which l



The reason why it's called U-Net and more detailed explanation of the architecture is entailed in this [documentation](https://obj-seg-doc-e3wipu72g6lsyt3rvkxp2g.streamlit.app/).

I also have made a well-detailed documentation, check it out [here](https://obj-seg-doc-e3wipu72g6lsyt3rvkxp2g.streamlit.app/).


