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
from sklearn.model_selection import train_test_split
import PIL
from PIL import Image
import time
import tensorflow as tf
import keras
from keras.layers import Input,Dense,Conv2D,concatenate,Dropout,LSTM
from keras import Model
from keras import activations
import warnings
warnings.filterwarnings("ignore")
import nltk.translate.bleu_score as bleu

from google.colab import drive 
drive.mount('/content/drive')
os.chdir("/content/drive/MyDrive/Medical_report_generation_2023")
from keras.applications import DenseNet121
image_shape= (224,224,3)
image_input= Input(shape=(224,224,3))
base=DenseNet121(include_top=False,input_tensor=image_input,input_shape=image_shape)
pred=Dense(14,"sigmoid")(base.output)
chexnet_model=Model(inputs=base.input,outputs=pred)
chexnet_model.load_weights("best_weights.h5")
chexnet_model.summary()
final_chexnet_model=Model(inputs=chexnet_model.inputs,outputs=chexnet_model.layers[-2].output,name="Chexnet_model")
train=pd.read_csv("train")
test=pd.read_csv("test")
leng=[]
for rep in train["report"]:
  leng.append(len(rep.split()))
print("90th percentile is ",np.percentile(leng,90))
print("99th percentile is ",np.percentile(leng,99))
print("max length is ",np.max(leng))
test.to_csv("test")
train.to_csv("train")
def image_feature_extraction(image1,image2):
  image_1 = Image.open(image1)
  image_1= np.asarray(image_1.convert("RGB"))
  image_2=Image.open(image2)
  image_2 = np.asarray(image_2.convert("RGB"))
  image_1=image_1/255
  image_2=image_2/255
  image_1 = cv2.resize(image_1,(224,224))
  image_2 = cv2.resize(image_2,(224,224))
  image_1= np.expand_dims(image_1, axis=0)
  image_2= np.expand_dims(image_2, axis=0)
  image_1_out=final_chexnet_model(image_1)
  image_2_out=final_chexnet_model(image_2)
  conc=np.concatenate((image_1_out,image_2_out),axis=2)
  image_feature=tf.reshape(conc, (conc.shape[0], -1, conc.shape[-1]))
  return image_feature

train_features=np.zeros((3056,98,1024))
test_features=np.zeros((764,98,1024))
for row in tqdm(range(train.shape[0])):
  image_1=train.iloc[row]["image1"]
  image_2=train.iloc[row]["image2"]
  train_features[row]=(image_feature_extraction(image_1,image_2))
for row in tqdm(range(test.shape[0])):
  image_1=test.iloc[row]["image1"] 
  image_2=test.iloc[row]["image2"]
  test_features[row]=(image_feature_extraction(image_1,image_2))
np.save("train_features_attention",train_features)
np.save("test_features_attention",test_features)
train_features=np.load("train_features_attention.npy")
test_features=np.load("test_features_attention.npy")
k=190
print(test_features[k])
one=test.iloc[k]["image1"] 
two=test.iloc[k]["image2"]
print(image_feature_extraction(one,two))
print(train_features.shape)
print(test_features.shape)
train_report=["<sos> "+text+" <eos>" for text in train["report"].values]
train_report_in=["<sos> "+text for text in train["report"].values]
train_report_out=[text+" <eos>" for text in train["report"].values]
test_report=["<sos> " +text+" <eos>" for text in test["report"].values]
test_report_in=["<sos> " +text for text in test["report"].values]
test_report_out=[text+" <eos>" for text in test["report"].values]
print(train_report_in[0])
print("*"*100)
print(train_report_out[0])
bs=10
max_len=80
token=tf.keras.preprocessing.text.Tokenizer(filters='' )
token.fit_on_texts(train_report)
vocab_size=len(token.word_index)+1 
seq=token.texts_to_sequences(train_report_in)
train_padded_inp=tf.keras.preprocessing.sequence.pad_sequences(seq,maxlen=max_len,padding="post")
seq=token.texts_to_sequences(train_report_out)
train_padded_out=tf.keras.preprocessing.sequence.pad_sequences(seq,maxlen=max_len,padding="post")
seq=token.texts_to_sequences(test_report_in)
test_padded_inp=tf.keras.preprocessing.sequence.pad_sequences(seq,maxlen=max_len,padding="post")
seq=token.texts_to_sequences(test_report_out)
test_padded_out=tf.keras.preprocessing.sequence.pad_sequences(seq,maxlen=max_len,padding="post")
embeddings_index=dict()
f = open('glove.6B.300d.txt')
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
print("Done")
embedding_matrix = np.zeros((vocab_size, 300))
for word, i in tqdm(token.word_index.items()):
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector
input_img=Input(shape=(98,1024),name="image_fetaures")
input_txt=Input(shape=(max_len),name="text_input")
en_out=Dense(enc_units,activation="relu",name="encoder_dense")(input_img)
enc_out=tf.keras.layers.Dropout(0.5)(en_out)
state1= Input(shape=(bs,enc_units),name="state1")
state2= Input(shape=(bs,enc_units),name="state2")
state_h=tf.keras.layers.Add()([state1,state2])
emb_out=tf.keras.layers.Embedding(vocab_size,output_dim=300,input_length=max_len,mask_zero=True,trainable=False,weights=[embedding_matrix])(input_txt)
weights=tf.keras.layers.AdditiveAttention()([state_h,en_out])
context_vector=tf.matmul(en_out,weights,transpose_b=True)[:,:,0]
context_vector=Dense(embedding_dim)(context_vector)
result=tf.concat([tf.expand_dims(context_vector, axis=1),emb_out],axis=1)
gru_out,state_1,state_2=tf.keras.layers.Bidirectional(tf.keras.layers.GRU(dec_units,return_sequences=True, return_state=True,name="Bidirectional_GRU"))(result)
out=tf.keras.layers.Dense(vocab_size,name="decoder_final_dense")(gru_out)
en_de=Model(inputs=[input_txt,input_img,state1,state2],outputs=out)
keras.utils.plot_model(en_de)
enc_units=64
embedding_dim=300
dec_units=64 
att_units=64
class Encoder(tf.keras.Model):
  def __init__(self,units):
    super().__init__()
    self.units=units
  def build(self,input_shape):
    self.dense1=Dense(self.units,activation="relu",name="encoder_dense")
    self.maxpool=tf.keras.layers.Dropout(0.5)
  def call(self,input_):
    enc_out=self.maxpool(input_)
    enc_out=self.dense1(enc_out) 
    return enc_out
  def initialize_states(self,batch_size):
      forward_h=tf.zeros((batch_size,self.units))
      back_h=tf.zeros((batch_size,self.units))
      return forward_h,back_h
class Attention(tf.keras.layers.Layer):
  def __init__(self,att_units):
    super().__init__()
    self.att_units=att_units
  def build(self,input_shape):
    self.wa=tf.keras.layers.Dense(self.att_units)
    self.wb=tf.keras.layers.Dense(self.att_units)
    self.v=tf.keras.layers.Dense(1)
  def call(self,decoder_hidden_state,encoder_output):
    x=tf.expand_dims(decoder_hidden_state,1)
    alpha_dash=self.v(tf.nn.tanh(self.wa(encoder_output)+self.wb(x)))
    alphas=tf.nn.softmax(alpha_dash,1)
    context_vector=tf.matmul(encoder_output,alphas,transpose_a=True)[:,:,0]
    return (context_vector,alphas)
class One_Step_Decoder(tf.keras.Model):
  def __init__(self,vocab_size, embedding_dim, input_length, dec_units ,att_units):
    super().__init__()
    self.att_units=att_units
    self.vocab_size=vocab_size
    self.embedding_dim=embedding_dim
    self.input_length=input_length
    self.dec_units=dec_units
    self.attention=Attention(self.att_units)
    self.embedding=tf.keras.layers.Embedding(self.vocab_size,output_dim=self.embedding_dim,
                                             input_length=self.input_length,mask_zero=True,trainable=False,weights=[embedding_matrix])
    self.gru= tf.keras.layers.Bidirectional(tf.keras.layers.GRU(self.dec_units,return_sequences=True, return_state=True))
    self.dense=tf.keras.layers.Dense(self.vocab_size,name="decoder_final_dense") 
    self.dense_2=tf.keras.layers.Dense(self.embedding_dim,name="decoder_dense2")
  def call(self,input_to_decoder, encoder_output, for_h,bac_h):
    embed=self.embedding(input_to_decoder)
    state_h=tf.keras.layers.Add()([for_h,bac_h])
    context_vector,alpha=self.attention(state_h,encoder_output)
    context_vector=self.dense_2(context_vector)
    result=tf.concat([tf.expand_dims(context_vector, axis=1),embed],axis=-1)
    output,forward_h,back_h=self.gru(result,initial_state=[for_h,bac_h])
    out=tf.reshape(output,(-1,output.shape[-1]))
    out=tf.keras.layers.Dropout(0.5)(out)
    dense_op=self.dense(out)
    return dense_op,forward_h,back_h,alpha
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, output_length, dec_units,att_units):
      super().__init__()
      self.onestep=One_Step_Decoder(vocab_size, embedding_dim, output_length, dec_units,att_units)
    def call(self, input_to_decoder,encoder_output,state_1,state_2):
        all_outputs=tf.TensorArray(tf.float32,input_to_decoder.shape[1],name="output_array")
        for step in range(input_to_decoder.shape[1]):
          output,state_1,state_2,alpha=self.onestep(input_to_decoder[:,step:step+1],encoder_output,state_1,state_2)
          all_outputs=all_outputs.write(step,output)
        all_outputs=tf.transpose(all_outputs.stack(),[1,0,2])
        return all_outputs
import warnings
warnings.filterwarnings("ignore")
class encoder_decoder(tf.keras.Model):
  def __init__(self,enc_units,embedding_dim,vocab_size,output_length,dec_units,att_units,batch_size):
        super().__init__()
        self.batch_size=batch_size
        self.encoder =Encoder(enc_units)
        self.decoder=Decoder(vocab_size,embedding_dim,output_length,dec_units,att_units)
  def call(self, data):
        features,report  = data[0], data[1]
        encoder_output= self.encoder(features)
        state_h,back_h=self.encoder.initialize_states(self.batch_size)
        output= self.decoder(report, encoder_output,state_h,back_h)
        return output

model  = encoder_decoder(enc_units,embedding_dim,vocab_size,max_len,dec_units,att_units,bs)
optimizer = tf.keras.optimizers.Adam()
loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='auto')
def custom_lossfunction(y_true, y_pred):
    mask = tf.math.logical_not(tf.math.equal(y_true, 0))
    loss_ = loss_function(y_true, y_pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ = loss_*mask
    loss_ = tf.reduce_mean(loss_)
    return loss_
model.compile(optimizer=optimizer,loss=custom_lossfunction)
red_lr=tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.2,patience=2, min_lr=0.0001) 
ckpt=tf.keras.callbacks.ModelCheckpoint("model2wts/ckpt",monitor='val_loss', verbose=0, save_best_only=True,save_weights_only=False, mode='auto')
model.fit([train_features[:3050],train_padded_inp[:3050]],train_padded_out[:3050],validation_data=([test_features[:760],test_padded_inp[:760]],test_padded_out[:760]),
          batch_size=bs,epochs=15,callbacks=[red_lr,ckpt])
model.load_weights("model2wts/bidir_fit_15_b")
def take_second(elem):
    return elem[1]
import time
def beam_search(image1,image2, beam_index):
    hidden_state =  tf.zeros((1, enc_units))
    hidden_state2 =  tf.zeros((1, enc_units))
    image_features=image_feature_extraction(image1,image2)
    encoder_out = model.layers[0](image_features)
    start_token = [token.word_index["<sos>"]]
    dec_word = [[start_token, 0.0]]
    while len(dec_word[0][0]) < max_len:
        temp = []
        for word in dec_word:
            predict, hidden_state,hidden_state2,alpha = model.layers[1].onestep(tf.expand_dims([word[0][-1]],1), encoder_out, hidden_state,hidden_state2)
            word_predict = np.argsort(predict[0])[-beam_index:]
            for i in word_predict:
                next_word, probab = word[0][:], word[1]
                next_word.append(i)
                probab += predict[0][i] 
                temp.append([next_word, probab.numpy()])
        dec_word = temp
        dec_word = sorted(dec_word, key=take_second)
        dec_word = dec_word[-beam_index:] 
    final = dec_word[-1]
    report =final[0]
    score = final[1]
    temp = []
    for word in report:
      if word!=0:
        if word != token.word_index['<eos>']:
            temp.append(token.index_word[word])
        else:
            break 
    rep = ' '.join(e for e in temp)        
    return rep, score
import random 
start=time.time()
i=random.sample(range(test.shape[0]),1)[0]
img1=test.iloc[i]["image1"]
img2=test.iloc[i]["image2"]
i1=cv2.imread(img1)
i2=cv2.imread(img2)
plt.figure(figsize=(10,6))
plt.subplot(131)
plt.title("image1")
plt.imshow(i1)
plt.subplot(132)
plt.title("image2")
plt.imshow(i2)
plt.show()
result,score=beam_search(img1,img2,3) 
actual=test_report[i]
print("ACTUAL REPORT: ",actual)
print("GENERATED REPORT: ",result)
end=time.time() 
print("BLEU SCORE IS: ",bleu.sentence_bleu(actual,result))
print("time required for the evaluation is ",end-start)
index=range(0,test.shape[0])
bl=0
start1=time.time()
for i in tqdm(index):
  img1=test.iloc[i]["image1"]
  img2=test.iloc[i]["image2"]
  result,sore=beam_search(img1,img2,3) 
  actual=test_report[i]
  bl+=bleu.sentence_bleu(actual,result)
end1=time.time()
print("\n")
print("average bleu score on the test data is ",bl/test.shape[0])
print("the average time for evaluating the attention model with beam search using bidirectinal GRU is ", (end1-start1)/764,"seconds")
