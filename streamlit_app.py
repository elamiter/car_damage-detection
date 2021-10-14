import streamlit as st
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import mobilenet
from PIL import Image
from skimage.transform import resize
import base64 


MODEL_PATHS=['./Saved_Models/Model_damage', './Saved_Models/Model_location', './Saved_Models/Model_severity']
@st.cache(allow_output_mutation=True)
def load_models(path):
    model = keras.models.load_model(path)
    return model

if __name__ == '__main__':

    # Designing the interface
    main_bg = "./Data/ArjangAutoCartoon.jpg"
    main_bg_ext = "jpg"
    st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
        no-repeat center center fixed;
        -webkit-background-size: cover;
        -moz-background-size: cover;
        -o-background-size: cover;
        background-size: cover;
    }}</style>
    """,
    unsafe_allow_html=True
    )
    st.title('Car-Damage-Classification-App with a CNN (transfer-learning)')
    st.write('\n')

    model_damage = load_models(MODEL_PATHS[0])
    model_location = load_models(MODEL_PATHS[1])
    model_severity = load_models(MODEL_PATHS[2])


    st.sidebar.title("Upload Image")

    #Disabling warning
    st.set_option('deprecation.showfileUploaderEncoding', False)
    #Choose your own image
    uploaded_file = st.sidebar.file_uploader(" ",type=['png', 'jpg', 'jpeg', 'JPEG', 'JPG'] )

    if uploaded_file is not None:
        
        u_img = Image.open(uploaded_file)
        st.image(u_img, 'Uploaded Image', use_column_width=True)
        # preprocess the image to fit the CNN model
        image = np.array(u_img)/127.5-1
        my_image= resize(image, (224,224)).reshape((1,224,224,3))
        


    st.sidebar.write('\n')
        
    if st.sidebar.button("Click Here to Classify"):
        
        if uploaded_file is None:
            
            st.sidebar.write("Please upload an Image to Classify")
        
        else:
            
            with st.spinner('Classifying ...'):
                
                prediction1 = model_damage.predict(my_image)
                prediction2 = model_location.predict(my_image)
                liste1 = ['front', 'rear', 'side']
                prediction3 = model_severity.predict(my_image)
                liste2 = ['minor', 'moderate', 'severe']
                st.success('Done!')
                
            st.sidebar.header("Algorithm Predicts: ")
            
        
            
            if prediction1[0][0] <= 0.5:
                
                st.sidebar.write("Car is damaged", '\n' )
                location = np.argmax(prediction2[0])
                st.sidebar.write("Car is damaged at this location of your car: ", liste1[location], '\n' ) 
                severity = np.argmax(prediction3[0])  
                st.sidebar.write("Severity of the damage: ", liste2[severity], '\n' )              
            else:
                st.sidebar.write("Are you sure your car has any damages?",'\n' )
                st.sidebar.write('if you are sure: try to zoom in/out and take a pic from a different angle')
