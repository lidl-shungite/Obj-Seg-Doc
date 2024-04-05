import streamlit as st
import time

def stream_line(sentence, sleep_time=0.02):
    for word in sentence.split():
        yield word + " "
        #time.sleep(sleep_time)

def main():
    st.page_link("pages/Documentation.py", label='')
    st.header("Documentaiton for Entire Process")
    container = st.container(border=False)
    container.subheader('Data :blue[Searching] and :green[Collection]')
    container.write_stream(stream_line('After deciding to use YOLOv8\'s backbone or architecture,'
    'and to train the pretrained weights, searching for the most suitable data came in priority. After a while of researching,'
    ' it is concluded that Pascal VOC dataset suits the best.'))
    container.write_stream(stream_line('Check out the datasets on [Roboflow](https://public.roboflow.com/object-detection/pascal-voc-2012)' 
                                       ' and [Kaggle](https://www.kaggle.com/datasets/gopalbhattrai/pascal-voc-2012-dataset).'))
    container.divider()
    container.subheader('Data :blue[Understanding] and :green[Preprocessing]')
    container.write_stream(stream_line("Next thing that came to mind was to unravel the data and try to understand the structure of data. "
                                       " The dataset has a total of 20 categories. "
                                       "Once, we realized that the data contained the feature image or original image along with its mask. "
                                       "Though, the mask is binary, it is not black and white, instead it is in black and a different color "
                                       "for different object. One more important thing is that YOLOv8 needs each individual label to be in a .txt format."
                                       " After a thorough research, we finally figured it out thanks to a YouTube video."))
    container.write_stream(stream_line('Check out his video on [YouTube](https://www.youtube.com/watch?v=aVKGjzAUHz0)' 
                                       ' and his [GitHub](https://github.com/computervisioneng/image-segmentation-yolov8) repository.'))
    container.divider()
    container.subheader("Transfering YOLOv8 :blue[Model]")
    container.write_stream(stream_line("Firstly, we install :blue[ultralytics] library that contains YOLO models."))
    pip_in = '''pip install ultralytics'''
    container.code(pip_in, language='python')
    container.write_stream(stream_line("Next we import the library and download the yolo model by running the code below."))
    ult_imp = '''
from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO('yolov8n-seg.yaml')

# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8n-seg.pt')
                
'''
    container.code(ult_imp, language='python')
    container.write_stream(stream_line("After downloading a model, it's trained with modified Pascal VOC dataset by passing the :blue[config.yaml]"
                                       " file that contains the total number of classes in the dataset along with their names and the paths to training and validation image folders."))
    train_mod_c = '''
    #Set and adjust epoch and imgsz as to your need
    results = model.train(data='config.yaml', epochs=100, imgsz=640)''' 
    container.code(train_mod_c, language='python')
    container.write_stream(stream_line("The next step would be building an application out of this model."))
    container.divider()
    container.header("")
if __name__ == '__main__':
    main()