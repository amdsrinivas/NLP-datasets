import streamlit as st

st.sidebar.title('Machine learning model trained on Stanford sentiment treebank.')

input_text = st.sidebar.text_area("Enter the sentence :")

submit_button = st.sidebar.button('Run the classifier')

if submit_button:
    # run the classifier over here.
    st.write('Need to implement!')