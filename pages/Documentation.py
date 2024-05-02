import streamlit as st
import time
from streamlit_image_comparison import image_comparison
from sklearn.metrics import confusion_matrix
import pandas as pd
import plotly.graph_objects as go

classNames =  ["person","bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush", "others"]    

def stream_line(sentence, sleep_time=0.02):
    for word in sentence.split():
        yield word + " "
        #time.sleep(sleep_time) 

def create_plot():
    df = pd.read_csv("logs/yolo_testing_data.csv")
    true = list(df["y_true"])
    pred = list(df["y_pred"])
    cm = confusion_matrix(y_true=true,y_pred=pred)
    fig = go.Figure(data=go.Heatmap(
    z=cm,
    x=classNames,
    y=classNames,
    colorscale='Viridis'))

    fig.update_layout(title='Confusion Matrix',
                    xaxis_title='Predicted',
                    yaxis_title='True')

    return fig

def main():
    lang_opt = st.sidebar.selectbox("Choose language: ",('English', '日本語'))
    st.page_link("pages/Documentation.py", label='')
    doc_opt = st.sidebar.selectbox("Choose documentation: ",("YOLOv8","U-Net"),placeholder="Choose documentation:")
    
    if lang_opt == "English":
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
            container.subheader("Data Analysis")
            container.plotly_chart(create_plot())
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
            container.subheader("Comparison of Original Image and Output of U-Net")
            img_opt = container.selectbox("Select an option:",("A Single Cat","Cat & Human"))
            if img_opt == "A Single Cat":
                image_comparison(img1="images/catge.jpg",img2="images/catge_predict.jpg",label1="Original Image",label2="Segmented Image", make_responsive=True)
            elif img_opt == "Cat & Human":
                image_comparison(img1="images/catnhuman.webp",img2="images/catnhuman.png",label1="Original Image",label2="Segmented Image", make_responsive=True)
    elif lang_opt == "日本語":
        st.header("Documentation for Entire Process")
        container = st.container(border=False)
        container.subheader('データの:blue[検索]と:green[収集]')
        container.write_stream(stream_line('YOLOv8 のバックボードやアーキテクチャを使用することを決定した後'
                                           '事前にトレーニングされたをトレーニングした後、最適なテーターを探'
                                           'すことが優先されます。しばらくの間、調査を行った結果、PascalVOC '
                                           'データーセットが最も適したと済まされます。'))
        container.write_stream(stream_line('[Roboflow](https://public.roboflow.com/object-detection/pascal-voc-2012) と ' 
                                        '[Kaggle](https://www.kaggle.com/datasets/gopalbhattrai/pascal-voc-2012-dataset) のホームページでデータセットをチェックしてください。'))
        container.divider()
        container.subheader('データ:blue[理解]と:green[前処理]')
        container.write_stream(stream_line("次に思い浮かんだものは、データ解読し、データの構造を理解しようとすることでした。"
                                           "このデータセットに合計２０カテゴリーがあります。一度、データには元の画像またはオ"
                                           "リジナル画像とそのマスクが含まれていることに気付きました。ただし、マスクは2値ですが、"
                                           "白黒ではなく、かわりに黒と別の色に異なります。異なるオブジェクトのために、"
                                           "もう一つ大切なのはYOLOv8が各個別のラベルを.txt形式で必要とすることです。念入りに研究す"
                                           "ることの結果の後、Youtube の動画のおかげで最終的にそれを理解することができます。"))
        container.write_stream(stream_line('[Youtube](https://www.youtube.com/watch?v=aVKGjzAUHz0) でこの動画をチェックして、この [GitHub](https://github.com/computervisioneng/image-segmentation-yolov8) リポジトリも見てみてください。'))
        container.divider()
        container.subheader("YOLOv8 :blue[モデル] の転送")
        container.write_stream(stream_line("まず、YOLO モデルを含む :blue[ultralytics] ライブラリを取り付けます。"))
        pip_in = '''pip install ultralytics'''
        container.code(pip_in, language='python')
        container.write_stream(stream_line("次に以下のコードを実行してライブラリをインポートし、YOLOモデルをダウロードします。"))
        ult_imp = '''
    from ultralytics import YOLO

    # ゼロから新しい YOLO モデルを作成する
    model = YOLO('yolov8n-seg.yaml')

    # 事前トレーニング済みの YOLO モデルを読み込む（トレーニングにおすすめ）
    model = YOLO('yolov8n-seg.pt')
                    
    '''
        container.code(ult_imp, language='python')
        container.write_stream(stream_line("モデルをダウロードした後、:blue[config.yaml] ファイルを渡して、修正した PascalVOC データセットでモデルをトレーニングします。"
                                           "このファイルでは、データセット内のクラスの総数と名前、及びトレーニングと検証の画像フォルンだーへのパスが含まれています。"))
        train_mod_c = '''
        # 必要に応じて、エポック数と画像サイズを設定して調整する
        results = model.train(data='config.yaml', epochs=100, imgsz=640)''' 
        container.code(train_mod_c, language='python')
        container.write_stream(stream_line("次のステップは、このモデルを使ってアプリケーションを構築することです。"))
        container.divider()
        container.subheader("元画像と YOLO の出力の比較")
        img_opt = container.selectbox("オプションを選択してください：",("一匹の猫","猫と人間"))
        if img_opt == "一匹の猫":
            image_comparison(img1="images/catge.jpg",img2="images/catge_predict.jpg",label1="オリジナル写真",label2="セグメンテッド写真", make_responsive=True)
        elif img_opt == "猫と人間":
            image_comparison(img1="images/catnhuman.webp",img2="images/catnhuman.png",label1="オリジナル写真",label2="セグメンテッド写真", make_responsive=True)

if __name__ == '__main__':
    main()