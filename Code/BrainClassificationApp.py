import streamlit as st
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
sns.set()
import modelHelper as helper
from PIL import Image
from os.path import exists
#from modelHelper import *
st.title('Brain Tumor Classifier')


uploaded_file = st.file_uploader("Upload Image")
    
if uploaded_file is not None:
    

    path = os.path.join('/Users/gordid/Desktop/MSAI/FAI/FinalProject/Data/TrainTestVal/Test/Normal', 
                        uploaded_file.name)
    
    file_exists = exists(path)
    
    if file_exists == False:
        
        path = os.path.join('/Users/gordid/Desktop/MSAI/FAI/FinalProject/Data/TrainTestVal/Test/Tumor', 
                            uploaded_file.name)


    # display the file
    display_image = Image.open(uploaded_file)
    display_image = display_image.resize((500,300))
    st.image(display_image)
    prediction = helper.predictImage(path)
    print(prediction)
    #os.remove('uploaded/'+uploaded_file.name)
    # drawing graphs
    st.text('Predictions :-')
    fig, ax = plt.subplots()
    ax  = sns.barplot(y = 'name',x='values', data = prediction)
    ax.set(xlabel='Confidence %', ylabel='Class')

    st.pyplot(fig)
        

    
    
#streamlit==1.13.0
#tensorflow==2.10.0
#tensorboard==2.10.1