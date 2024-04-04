import streamlit as st
import time


def stream_line(sentence, sleep_time=0.02):
    for word in sentence.split():
        yield word + " "
        time.sleep(sleep_time)


def main():
    st.page_link("pages/Documentation.py", label='')
    st.header('Documentation of :blue[Object Segmentation AI] using :green[YOLOv8]')
    container = st.container(border=False)
    container.write_stream(stream_line('There are mainly three types of segmentation: Semantic, Instance, and Panoptic. '
                                       'This model in particular is about semantic segmentation. More of this on the documentation page.'))
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