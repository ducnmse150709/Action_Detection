#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Importing libraries
import csv
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import image as img

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

import os
import random
from PIL import Image
import sys
from tqdm.notebook import tqdm
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from keras import layers
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator


# In[9]:


# Reading the csv file into a dataframe

df = pd.read_csv('../input/Human Action Recognition/Training_set.csv')
df


# In[4]:


# Checking the data types

df.dtypes


# In[ ]:


# Checking for null values

df.isnull().sum()


# In[5]:


# Checking label counts

counts = df['label'].value_counts()
counts


# In[6]:


# Function to display few random images

def displayRandom(n=1):
    plt.figure(figsize=(20,20))
    for i in range(n):
        rnd = random.randint(0,len(df)-1)
        img_file = '../input/Human Action Recognition/train/' + df['filename'][rnd]

        if os.path.exists(img_file):
            plt.subplot(n//2+1, 2, i + 1)
            image = img.imread(img_file)
            plt.imshow(image)
            plt.title(df['label'][rnd])


# In[9]:


# Displaying 4 random images with corresponding label

displayRandom(4)


# In[16]:


# Label encoding and seperate dependant variable

lb = LabelBinarizer()
y = lb.fit_transform(df['label'])
classes = lb.classes_
print(classes)


# In[17]:


# Take independant variable as numpy array

x = df['filename'].values


# In[18]:


# Split data as 90% of training and 10% of test data

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=100) 


# In[19]:


# Load pixel data as a list of numpy arrays

img_data = []
size = len(x_train)

for i in tqdm(range(size)):
    image = Image.open('../input/Human Action Recognition/train/' + x_train[i])
    img_data.append(np.asarray(image.resize((160,160))))


# In[11]:


# Creating the model 

model = Sequential()

pretrained_model= `,
                   input_shape=(160,160,3),
                   pooling='avg',classes=15,
                   weights='imagenet')

for layer in pretrained_model.layers:
        layer.trainable=False

model.add(pretrained_model)
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(15, activation='softmax'))


# In[12]:


model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()


# In[ ]:


# Training the model

history = model.fit(np.asarray(img_data), y_train, epochs=60)


# In[ ]:


# Saving the model weights

model.save_weights("model.h5")


# In[ ]:


# Accuracy graph

accu = history.history['accuracy']
plt.plot(accu)


# In[ ]:


# Losses graph

losss = history.history['loss']
plt.plot(losss)


# In[ ]:


# Model accuracy

test_img_data = []
size = len(x_test)

for i in tqdm(range(size)):
    image = Image.open('D:\Human Action Recognition/train/' + x_test[i])
    test_img_data.append(np.asarray(image.resize((160,160))))

scores = model.evaluate(np.asarray(test_img_data), y_test)
print(f"Test Accuracy: {scores[1]}")


# In[ ]:


# Function to read images as array

def read_image(fn):
    image = Image.open(fn)
    return np.asarray(image.resize((160,160)))


# In[ ]:


# Function to predict

def test_predict(test_image):
    result = model.predict(np.asarray([read_image(test_image)]))

    itemindex = np.where(result==np.max(result))
    prediction = classes[itemindex[1][0]]
    print("probability: "+str(np.max(result)*100) + "%\nPredicted class : ", prediction)

    image = img.imread(test_image)
    plt.imshow(image)
    plt.title(prediction)


# In[ ]:


test_predict('D:\Human Action Recognition/test/Image_152.jpg')


# In[29]:


test_predict('D:\Human Action Recognition/test/Image_3305.jpg')


# In[22]:


test_predict('D:\Human Action Recognition/test/Image_3300.jpg')


# In[24]:


test_predict('D:\Human Action Recognition/test/Image_2300.jpg')


# In[25]:


test_predict('D:\Human Action Recognition/test/Image_300.jpg')


# In[30]:


test_predict('D:\Human Action Recognition/test/Image_1.jpg')


# In[1]:


test_predict('D:\Human Action Recognition/test/Image_2.jpg')


# In[32]:


test_predict('D:\Human Action Recognition/test/Image_3.jpg')


# In[33]:


test_predict('D:\Human Action Recognition/test/Image_4.jpg')


# In[ ]:




