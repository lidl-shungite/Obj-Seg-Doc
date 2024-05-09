import streamlit as st
import time


def stream_line(sentence, sleep_time=0.02):
    for word in sentence.split():
        yield word + " "
        time.sleep(sleep_time)


def main():
    st.sidebar.page_link("pages/Documentation.py", label='')
    st.sidebar.selectbox("Choose language: ",('English', '日本語'))
    
    st.header('Introduction to Object :blue[Segmentation]')
    container = st.container(border=False)
    container.write_stream(stream_line('What is Object Segmentation? Object segmentation is a critical task in computer vision that involves identifying '
                                       'and delineating objects within an image or video. This process is essential for various applications, including '
                                       'autonomous vehicles, medical imaging, and surveillance systems.'))
    container.write_stream(stream_line(':blue[Semantic] Segmentation: Assigns a class label to every pixel in an image, treating multiple objects of the same class as a single entity.'))
    container.write_stream(stream_line(':blue[Instance] Segmentation: Identifies and labels each individual instance of an object within an image, even if they belong to the same class.'))
    container.write_stream(stream_line(':blue[Panoptic] Segmentation: Combines semantic and instance segmentation by assigning both a semantic label and a unique instance identifier to each pixel, providing a comprehensive view of the scene.'))
    #container.image("other_images/phone_calc-removebg-preview.png")

if __name__ == '__main__':
    main()