import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow import keras
import keras.backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array


def f1_score(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0.0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


modelPath = '/Users/gordid/Desktop/MSAI/FAI/FinalProject/ENB4_BrainTumorClassifier.h5'
model = keras.models.load_model(modelPath,
                               custom_objects={"f1_score": f1_score})


def predictImage(imagePath):
    
    img = load_img(imagePath,target_size=(300,300,3))
    img_array = np.array(img)
    print(img_array.shape)
    img_array= img_array[np.newaxis, :]
    
    prediction = model.predict(img_array)*100
    prediction = pd.DataFrame(np.round(prediction,1),columns = ["Normal", "Tumor"]).transpose()
    prediction.columns = ['values']
    #prediction  = prediction.nlargest(5, 'values')
    prediction = prediction.reset_index()
    prediction.columns = ['name', 'values']
    return(prediction)


#predictImage('/Users/gordid/Desktop/MSAI/FAI/FinalProject/Data/TrainTestVal/Test/Normal/DS1_T0_no0.jpg')
# =============================================================================
# print('model loaded')
# def predictor(img_path): # here image is file name 
#     # base_path = os.path.join(current_path, 'static\images\cache')
#     # path = os.path.join(base_path,image_name)
#     img = load_img(img_path, target_size=(331,331))
#     # print(path)
#     # img = cv2.resize(img,(331,331))
#     img = img_to_array(img)
#     img = np.expand_dims(img,axis = 0)
#     features = feature_extractor.predict(img)
#     prediction = predictor_model.predict(features)*100
#     prediction = pd.DataFrame(np.round(prediction,1),columns = dog_breeds).transpose()
#     prediction.columns = ['values']
#     prediction  = prediction.nlargest(5, 'values')
#     prediction = prediction.reset_index()
#     prediction.columns = ['name', 'values']
#     return(prediction)
# =============================================================================

    


# print(predictor('samoyed_puppy_dog_pictures.jpg'))


