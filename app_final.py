
import flask
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import sklearn
import tqdm
from tqdm import tqdm 
import nltk
import warnings
warnings.filterwarnings("ignore") 
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array
from keras.applications.vgg19 import preprocess_input
from sklearn.model_selection import train_test_split
import PIL
from PIL import Image
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input,Dense,Conv2D,concatenate,Dropout,LSTM
from tensorflow.keras import Model
from tensorflow.keras import activations
import warnings
warnings.filterwarnings("ignore")
import nltk.translate.bleu_score as bleu
from keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import cv2
from PIL import Image,ImageOps
from PIL import ImageDraw 
import shutil
import os
from PIL import Image
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array
from keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Input, Embedding, Conv2D, Concatenate, Flatten, Add, Dropout, GRU
import random
import datetime
from nltk.translate.bleu_score import sentence_bleu

app = Flask(__name__)

# Custom loss function
loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction='auto')
def maskedLoss(y_true, y_pred):
    mask = tf.math.logical_not(tf.math.equal(y_true, 0))
    loss_ = loss_function(y_true, y_pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ = loss_*mask
    loss_ = tf.reduce_mean(loss_)
    return loss_

encoder_decoder=load_model("D:\Krishna\ML\model.h5",custom_objects={'maskedLoss':maskedLoss})
encoder_decoder.compile(optimizer="Adam", loss = maskedLoss)

#Loading the pretrained tokenizer
with open(r'D:\Krishna\ML\token.pickle', 'rb') as handle:
    token= pickle.load(handle) 

def img_preprocess(img):
    img=img_to_array(img)   
    img=preprocess_input(img)
    img=cv2.resize(img,(224,224))
    img=img/255.0
    img=np.expand_dims(img, axis=0)
    return img

def evaluation(image1,image2):
    image1 = Image.open(image1)
    image2 = Image.open(image2)
    
    image1.show()
    image2.show()
    
    image1=img_preprocess(image1)
    image2=img_preprocess(image2)

    from tensorflow.keras.applications import DenseNet121

    image_shape= (224,224,3)
    mod2=DenseNet121(include_top=False,input_shape=image_shape,pooling="avg")
    las2=Dense(14,"sigmoid")(mod2.output)

    mod2=Model(inputs=mod2.input,outputs=las2)
    mod2.load_weights(r"D:\Krishna\ML\chexnet_weights.h5")
    
    final_chexnet_model=Model(inputs=mod2.inputs,outputs=mod2.layers[-2].output,name="Chexnet_model")
    
    image_1= tf.keras.Input(shape=(224,224,3),name="image_1_features")
    image_2= tf.keras.Input(shape=(224,224,3),name="image_2_features")
    image_1_out=final_chexnet_model(image_1)
    image_2_out=final_chexnet_model(image_2)
    conc=tf.keras.layers.Concatenate(axis=-1)([image_1_out,image_2_out])
    feature_extraction_model=Model(inputs=[image_1,image_2],outputs=conc)
    
    image_features=feature_extraction_model([image1,image2])
    output_report=''
    input_rep= 'startseq'
    image_features=tf.reshape(image_features,shape=(-1,image_features.shape[-1]))
    
    for i in range(80):
        input_tokens = [token.word_index[w] for w in input_rep.split()]
        input_padded = tf.keras.preprocessing.sequence.pad_sequences([input_tokens],80, padding='post')
        results = encoder_decoder.predict([image_features,input_padded])
        arg = np.argmax(results[0]) 
        if token.index_word[arg]=='endseq':
            output_report+=" "
            break
        else:
            input_rep = input_rep + ' ' + token.index_word[arg]
            output_report = output_report+token.index_word[arg]+" "
    return output_report

  
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    image_link = [x for x in request.form.values()]
    
    # final_features = feature_extraction(image_link[0],image_link[1])
    prediction = evaluation(image_link[0],image_link[1])
    
    
    return render_template('predict.html', prediction_text=prediction)
'''
@app.route('/predict_api',methods=['POST'])
    def predict_api():
    
    For direct API calls trought request
    
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)
    '''

if __name__ == "__main__":
    app.run(debug=False,threaded=False)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    