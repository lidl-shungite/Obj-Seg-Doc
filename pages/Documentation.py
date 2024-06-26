import streamlit as st
import time
from streamlit_image_comparison import image_comparison
from sklearn.metrics import confusion_matrix
import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px

classNames =  ["aeroplane","bicycle", "bird", "boat","bottle","bus","car", "cat", "chair", "cow", "diningtable", "dog", "horse",
              "motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"]    

def stream_line(sentence, sleep_time=0.02):
    for word in sentence.split():
        yield word + " "
        #time.sleep(sleep_time) 

def create_plot(df, mode):
    fig = px.line(data_frame=df, x='Epoch', y=[f'Training {mode}', f'Validation {mode}'], hover_name='Epoch',
                     labels={'Epoch': 'Epochs', 'value': f'{mode}', 'variable': 'Dataset'},
                     color_discrete_map={f'Training {mode}': '#1f77b4', f'Validation {mode}': '#ff7f0e'})
    fig.update_layout(title=f"Graph for {mode}",xaxis=dict(showgrid=True), yaxis=dict(showgrid=True), title_font=dict(size=16))
    return fig

def plot_confusion_matrix():
    df = pd.read_csv("logs/yolo_testing_data.csv")
    cm = confusion_matrix(y_true=list(df["y_true"]), y_pred=list(df["y_pred"]))
    colorscale = [[0.0, '#f5f5f5'], [0.5, '#1f77b4'], [1.0, '#ff7f0e']]
    fig = ff.create_annotated_heatmap(z=cm, x=classNames, y=classNames, colorscale=colorscale)
    fig.update_layout(title="Confusion Matrix",xaxis=dict(title='Predicted Label'), yaxis=dict(title='True Label'), title_x=0.485,
        title_y=0.075, title_font=dict(size=16))
    
    return fig

def main():
    st.page_link("pages/Documentation.py", label='')
    doc_opt = st.sidebar.selectbox("Choose documentation: ",("YOLOv8","U-Net"),placeholder="Choose documentation:")

    if doc_opt == "YOLOv8":
        st.header(f"Documentation for :blue[{doc_opt}]")
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
                                        " file that contains the total number of classes in the dataset along with their names and the paths to training and validation image folders."
                                        " The parameter :blue[imgsz] is given to set the size of images during training to be the same."))
        train_mod_c = '''
        #Set and adjust epoch and imgsz as to your need
        results = model.train(data='config.yaml', epochs=100, imgsz=640)''' 
        container.code(train_mod_c, language='python')
        container.write_stream(stream_line("The next step would be building an application out of this model."))
        container.divider()
        container.subheader("Testing :blue[YOLOv8] model")
        container.write_stream(stream_line("We collected a total of 100 images for 20 classes, and tested out to see how well would YOLOv8 performs."
                                           " The model has an overall accuracy of 52.2:blue[%]. The following is a confusion matrix to show how correct"
                                           " is our model when it comes to particular classes."))
        container.plotly_chart(plot_confusion_matrix())
        container.divider()
        container.subheader("Comparison of Original Image and Output of YOLO")
        img_opt = container.selectbox("Select an option:",("A Single Cat","Cat & Human"))
        if img_opt == "A Single Cat":
            image_comparison(img1="images/catge.jpg",img2="images/catge_predict.jpg",label1="Original Image",label2="Segmented Image", make_responsive=True)
        elif img_opt == "Cat & Human":
            image_comparison(img1="images/catnhuman.webp",img2="images/catnhuman.png",label1="Original Image",label2="Segmented Image", make_responsive=True)
    
    elif doc_opt == "U-Net":
        st.header(f"Documentation for :blue[{doc_opt}]")
        container = st.container(border=False)
        container.subheader("Introducing :blue[U-Net]")
        container.write_stream(stream_line('After searching around the Internet forums for a while,'
                                           ' most of them kept mentioning the infamous Recurrent Neural Network Architecture from 2015, '
        'called U-Net. One thing about U-Net is that it was initially intended for Medical Image Segmentation, but since then, it has been'
        ' used for segmentation of other things as well. Alas, U-Net has been chosen for our project. We will come back to why it is named \"U-Net\".'))
        container.divider()
        container.subheader('Data :blue[Searching] and :green[Collection]')
        container.write_stream(stream_line('I wanted to see how our custom trained model would perform on the dataset that I\'ve trained on YOLO. '
                                           'So, it was decided that we would use PascalVOC for training U-Net.'))
        container.write_stream(stream_line('Check out the datasets on [Roboflow](https://public.roboflow.com/object-detection/pascal-voc-2012)' 
                                        ' and [Kaggle](https://www.kaggle.com/datasets/gopalbhattrai/pascal-voc-2012-dataset).'))
        container.divider()
        container.subheader('Data :blue[Understanding] and :green[Preprocessing]')
        container.write_stream(stream_line("We wanted to make the U-Net model outputs binary mask, so that it would be much better off,"
                                           " when comparing two models. So, I set out to process the mask images which are in black or "
                                           "in some other different color, to be solely in black and white. This was easily achieved by "
                                           "normalizing the pixel values of the mask image, by simply dividing it with :blue[255.0]. "
                                           "This process made the pixel values to be between :blue[0] and :blue[1]. The values are then multiplied by :green[255] "
                                           "to make the binary mask."))
        container.write_stream(stream_line('Check out the referenced source code from this [YouTube](https://youtu.be/n4_ZuntLGjg?si=7zq-8cpIoaHgSDKN) video.'))
        container.divider()
        container.subheader("Understanding :blue[U-Net]")
        container.image('images/unet_arch.png', caption='Architecture of U-Net (credits to the owner)')
        container.write_stream(stream_line("U-Net is a Recurrent Neural Network architecture, which uses encoders and decoders."
                                           " In the provided picture, blocks that are named \":blue[conv]\", are the encoders and their"
                                           "purpose is to extract crucial features, otherwise known as \"Down-Sampling\""
                                           ". The blocks named \":blue[deconv]\" are decoders, also known as \"Up-Sampling Layers\""
                                           " which are used to rebuild the de-sampled down images. At the start of a segmentation process, the data is passed "
                                           "along the encoding layers, down-sampling them. :red[Notice]: _Copies of the down-sampled values are send to the decoded"
                                           " blocks to skip connections, where later they will be concatted upon the up-sampled values._ The down-sampling process "
                                           "escalates further shrinking the size of data; this cuts out unimportant or insignificant features and keeps the crucial parts."
                                           " The outputs from the down-sampling layers are up-sampled to reach the original data size."))
        
        
        container.divider()
        container.subheader("Training and Validating the Model")
        container.write_stream(stream_line("After building the model, we now commence to train and validate the model "
                                            "using the data. The data is split into 9-part training and 1-part validation, "
                                            "meaning 90% of data is poured into training and 10% into validation. The following "
                                            "are the two graphs representing the learning curve; one for the accuracy and the other "
                                            "for the loss. The learning curve shows how our model's performance changes after each " 
                                            "epoch. Here, loss(error), MSE(Mean Squared Error) and accuracy are three metrics used to measure the performance."
                                            " It is remarked that the model is overfitting, since the gap between two lines gradually"
                                            " becomes larger every epoch. This applies for every graph there is. The epoch counts stopped "
                                            "at 41, this occured because Early Stopping was used to halt the training process, if the"
                                            " validation loss doesn't improve for about 25 epochs. "))

        tab1, tab2, tab3 = container.tabs(["Learning Curve ( :green[Accuracy] )", "Learning Curve ( :blue[Loss] )", "Learning Curve ( :red[Mean Squared Error])"])
        with tab1:
            tab1.plotly_chart(create_plot(pd.read_csv("logs/acc_log.csv"), 'Accuracy'), theme="streamlit", use_container_width=True)
        with tab2:
            tab2.plotly_chart(create_plot(pd.read_csv("logs/loss_log.csv"), 'Loss'), theme="streamlit", use_container_width=True)
        with tab3:
            tab3.plotly_chart(create_plot(pd.read_csv("logs/mse_log.csv"), 'MSE'), theme="streamlit", use_container_width=True)
        container.divider()
        container.subheader("Comparison of Original Image and Output of U-Net")        
        image_comparison(img1="images/dog-2-resize.jpg",img2="images/dog2_with_mask.jpg",label1="Original Image",label2="Segmented Image", make_responsive=True)
if __name__ == '__main__':
    main()

