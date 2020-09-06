#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

from tensorflow.python.keras.models import Sequential
from tensorflow.keras.preprocessing.image import load_img, img_to_array 
from tensorflow.python.keras.layers import Dense, Flatten, AveragePooling2D, Conv2D, MaxPooling2D,Activation,Dropout,Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer

from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os


# In[2]:


dataset=r"C:\Users\hp\OneDrive\Desktop\dataset of face mask detection\data"
print(os.listdir(dataset))


# In[5]:


data=[]
labels=[]
for category in os.listdir(dataset):
    path=os.path.join(dataset,category)
    if not os.listdir(path):
        continue
    for image in os.listdir(path):
        if image.startswith('.'):
            continue
        img=cv2.imread(os.path.join(path,image))
# #         print(img)
        try:
            img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img=cv2.resize(img,(110,110))
            
#             img=load_img(os.path.join(path,image),target_size=(224,224))
#             img=img_to_array(img)
#             img=preprocess_input(img)
            data.append(img)
            labels.append(category)
        except Exception as e:
            print('Exception ',e)
  
                     
         
        


# In[6]:


data=np.array(data)/255.0
data=np.reshape(data,(data.shape[0],110,110,1))
data.shape

lb=LabelBinarizer()
labels=lb.fit_transform(labels)
labels=to_categorical(labels)
labels=np.array(labels)


# In[7]:


model=Sequential()
model.add(Conv2D(200,(3,3),input_shape=(110,110,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#The first CNN layer followed by Relu and MaxPooling layers

model.add(Conv2D(100,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#The second convolution layer followed by Relu and MaxPooling layers

model.add(Flatten())
model.add(Dropout(0.5))
#Flatten layer to stack the output convolutions from second convolution layer
model.add(Dense(50,activation='relu'))
#Dense layer of 64 neurons
model.add(Dense(2,activation='softmax'))
#The Final layer with two outputs for two categories

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[8]:


# model = Model(inputs=basemodel.input, outputs=headModel)
model.summary()


# In[9]:


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[10]:


trainx,testx,trainy,testy=train_test_split(data,labels,test_size=0.1)


# In[11]:


checkpoint = ModelCheckpoint('mask_detection-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')
his=model.fit(trainx,trainy,epochs=20,callbacks=[checkpoint],validation_split=0.2)


# In[12]:


y_pred=model.predict(testx)
pred=np.argmax(y_pred,axis=1)
ground = np.argmax(testy,axis=1)
print(classification_report(ground,pred))


# In[13]:


get_acc = his.history['accuracy']
value_acc = his.history['val_accuracy']
get_loss = his.history['loss']
validation_loss = his.history['val_loss']

epochs = range(len(get_acc))
plt.plot(epochs, get_acc, 'r', label='Accuracy of Training data')
plt.plot(epochs, value_acc, 'b', label='Accuracy of Validation data')
plt.title('Training vs validation accuracy')
plt.legend(loc=0)
plt.figure()


plt.show()


# In[14]:


epochs = range(len(get_loss))
plt.plot(epochs, get_loss, 'r', label='loss of Training data')
plt.plot(epochs, validation_loss, 'b', label='loss of Validation data')
plt.title('Training vs validation loss')
plt.legend(loc=0)
plt.figure()


plt.show()


# In[20]:


prediction_image=cv2.imread(os.path.join(dataset,'with_mask','377-with-mask.jpg'))


# In[21]:


prediction_image.shape


# In[22]:


prediction_image=cv2.cvtColor(prediction_image,cv2.COLOR_BGR2GRAY)
prediction_image=cv2.resize(prediction_image,(110,110))
prediction_image=np.array(prediction_image)/255.0
pred
plt.imshow(prediction_image)


# In[23]:


prediction_image.shape


# In[24]:


prediction_image= np.reshape(prediction_image,(1,110,110,1))
prediction_image.shape


# prediction_image=np.expand_dims(prediction_image,axis=0)


# In[25]:


categories={0:'with_mask',1:'without_mask'}
def mapping(value):
    return categories[value]


# In[26]:



prediction=model.predict(prediction_image)
value=np.argmax(prediction)
predicted=mapping(value)
print("PREDICTION IS {}".format(predicted))


# In[27]:


# model.save("mask_detection.h5")
from tensorflow.keras.models import model_from_json


# In[28]:


# prediction_image.shape
json_file = model.to_json()
with open("mask_detection.json", "w") as file:
    file.write(json_file)
model.save_weights("mask_detection.h5")


# In[ ]:




