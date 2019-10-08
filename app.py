import streamlit as st
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow_hub as hub
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


matplotlib.use('Agg')
st.title('Serving a Keras Model')

model = tf.keras.models.load_model('sst-3-class.hd5')
preprocessor = hub.Module("D:\\Workspace\\NLP\\tfhub-modules\\google-universal-sentence-encoder")
init = tf.global_variables_initializer()
table_init = tf.tables_initializer()
# st.write(model)
a = None
input_sample = st.sidebar.text_area("Enter the input sentence :")
if st.sidebar.button("Submit"):
    a = st.text("You submitted : {0}".format(input_sample))
    features = preprocessor([input_sample])
    with tf.Session() as sess:
        sess.run([init, table_init])
        features = sess.run(features)
    outputs = model.predict(features)
    print(outputs)
    x_ticks = ['Negative', 'Neutral', 'Positive']
    # st.bar_chart(pd.DataFrame(data=outputs, columns=x_ticks) )
    y_pos = np.arange(len(x_ticks))

    probas = outputs
    plt.bar(y_pos, probas.tolist()[0], align='center', alpha=0.5)
    plt.xticks(y_pos, x_ticks)
    plt.ylabel("Probabilities")
    plt.title('Model Outputs')
    # plt.show()
    st.pyplot(plt)
    st.write("Model Prediction : {0}".format(x_ticks[np.argmax(outputs)]))

if st.sidebar.button("clear me"):
    a = None