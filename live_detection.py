#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import cv2
import os
import tensorflow
from tensorflow.keras.models import model_from_json,load_model
import matplotlib as mpl
import matplotlib.pyplot as plt


# In[2]:



loaded_model = load_model('mask_detection-008.model')


# In[3]:


categories={0:"with mask",1:"without-mask"}
def mapping(value):
    return categories[value]


# In[4]:


import tkinter
from tkinter import messagebox
import smtplib


# In[5]:


root=tkinter.Tk()
root.withdraw()


# In[6]:


face_cascade = cv2.CascadeClassifier('C:\\Users\\hp\\OneDrive\\Desktop\\dataset of face mask detection\\haarcascade_frontalface_default.xml')


# In[7]:


vid_src=cv2.VideoCapture(0)
rect_col={0:(0,255,0),1:(0,0,255)}

Subject='KHATRA'
MSG='Aaap ki jeevan khatre m hain,kripya mask phene anyatha aap bhagwan ko pyare ho jaoge'


# In[8]:


while(True):
    ret,img=vid_src.read()
    prediction_image=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(prediction_image,1.3,5)
    
    for (x,y,w,h) in faces:
        face_img=prediction_image[y:y+w,x:x+w]
        resized_img=cv2.resize(face_img,(110,110))
        normalised_img=resized_img/255.0
        reshaped_img=np.reshape(normalised_img,(1,110,110,1))
        result=loaded_model.predict(reshaped_img)
        label=np.argmax(result,axis=1)[0]
        
        cv2.rectangle(img,(x,y),(x+w,y+h),rect_col[label],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),rect_col[label],-1)
        cv2.putText(img,categories[label],(x,y-10),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),2)
        
        if(label==1):
            messagebox.showwarning("KHATRA","AAPKI JAAN SANKAT MAIN HAI")
            message='Subject: {}\n\n{}'.format(Subject,MSG)
            mail=smtplib.SMTP('smtp.gmail.com',587)
            mail.ehlo()
            mail.starttls()
            mail.login('ritikmeena.gwl.2000@gmail.com','enter ur pswd here')
            mail.sendmail('ritikmeena.gwl.2000@gmail.com','ritikmeena.gwl.2000@gmail.com',message)
            mail.quit()
        else:
            messagebox.showwarning("NO KHATRA","AAPKI JAAN SURAKSHIT HAI")
            pass
            break
    cv2.imshow('Live Video feed',img)
    key=cv2.waitKey(1)
    
    if(key==27):
        break
vid_src.release()
cv2.destroyAllWindows()

    
            

    


# In[ ]:




