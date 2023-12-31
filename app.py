import streamlit as st
st.set_page_config(page_title="Farm Smart", layout="wide")


st.title("Farm it Smart")
st.image('Leaf.png', width=100)
padding = 0
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)

image = 'Windmill.gif'
st.image(image=image, caption="Windmill")
st.write("Please navigate through the sections at the side.")
