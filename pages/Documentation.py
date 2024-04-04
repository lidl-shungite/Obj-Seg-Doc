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
    container.subheader('Data :blue[Understanding] and :green[Preprocessing]')
    container.write_stream(stream_line("Next thing that came to mind was to unravel the data and try to understand the structure of data. "
                                       "Once, we realized that the data contained the feature image or original image along with its mask. "
                                       "Though, the mask is binary, it is not black and white, instead it is in black and a different color "
                                       "for different object. One more important thing is that YOLOv8 needs each individual label to be in a .txt format."
                                       " After a thorough research, we finally figured it out thanks to a YouTube video."))
    container.write_stream(stream_line('Check out his video on [YouTube](https://www.youtube.com/watch?v=aVKGjzAUHz0)' 
                                       ' and his [GitHub](https://github.com/computervisioneng/image-segmentation-yolov8) repository.'))
    container.subheader("Transfering YOLOv8 :blue[Model]")
    container.write_stream(stream_line("Firstly, we install :blue[ultralytics] a library built by "))
    pip_in = '''pip install ultralytics'''
    container.code(pip_in, language='python')



    # container.write_stream(stream_line(
    #     ''
    #     ''))
    # container.write_stream(stream_line('No more scratching your head over math homework. With PicCalcBot, you can say '
    #     'goodbye to math stress and hello to quick and accurate solutions.'))
    # container.write_stream(stream_line('So why wait? Download PicCalcBot now and let the math magic begin! Math class just '
    #     'got a whole lot cooler!'))

    #container.image("other_images/phone_calc-removebg-preview.png")

if __name__ == '__main__':
    main()