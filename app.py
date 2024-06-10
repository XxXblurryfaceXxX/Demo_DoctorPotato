import time
import streamlit as st
import tensorflow as tf
import numpy as np

#Tensorflow Model Prediction

def mode_prediction(test_image):
    model = tf.keras.models.load_model('DoctorPotatoModel.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(256,256))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #Convert single image to a batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index


#Sidebar

st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])


#HomePage

if(app_mode == "Home"):
    st.header("Plant Disease Recognition System")


#Prediction Page

elif(app_mode == "Disease Recognition"):

    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=300)
    
    #Predict Button
    if(st.button("Predict")):
        with st.spinner("Por favor espera ..."):
            time.sleep(2)
            st.write("Our Prediction")
            resul_index = mode_prediction(test_image)
            #Define Class
            class_name = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
            st.success("Model is Predicting it's a {}".format(class_name[resul_index]))





