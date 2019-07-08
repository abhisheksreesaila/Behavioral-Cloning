
# coding: utf-8

# In[1]:


import csv
import cv2
import os
import zipfile as z
import numpy as np
import pandas as pd


# In[2]:


from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Activation, Reshape, Dropout, Cropping2D
from keras.layers import  Conv2D
from keras.layers import MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint


# In[4]:


# _zipfile = z.ZipFile('/home/carnd/BehavCloning/BPTrainigData.zip')
# _zipfile.extractall('/home/carnd/BehavCloning/')
# _zipfile.close()


# In[ ]:


#build
images = []
measurements = []


# In[23]:


lines = []

with open('/home/carnd/BehavCloning/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
        
del lines[0]   


for line in lines:
            
            ##############################################################
            #center image
            source_path = line[0]
            filename = source_path.split('/')[-1]
            current_path = '/home/carnd/BehavCloning/data/IMG/' + filename
            C_image = cv2.imread(current_path)
            C_image_CONVERTED = cv2.cvtColor(C_image, cv2.COLOR_BGR2RGB)
            
            
            #left image
            source_path = line[1]
            filename = source_path.split('/')[-1]
            current_path = '/home/carnd/BehavCloning/data/IMG/' + filename
            L_image = cv2.imread(current_path)
            L_image_CONVERTED = cv2.cvtColor(L_image, cv2.COLOR_BGR2RGB)
            
            #right image
            source_path = line[2]
            filename = source_path.split('/')[-1]
            current_path = '/home/carnd/BehavCloning/data/IMG/' + filename
            R_image = cv2.imread(current_path)
            R_image_CONVERTED = cv2.cvtColor(R_image, cv2.COLOR_BGR2RGB)  # converting to RGB since drive.py feeds the model that way
            ##############################################################
            
            # add to the images list
            images.append((C_image_CONVERTED))
            measurement = float(line[3])
            measurements.append(measurement)  
            
            images.append((L_image_CONVERTED))
            measurement = float(line[3]) + 0.2
            measurements.append(measurement)  
            
            images.append((R_image_CONVERTED))
            measurement = float(line[3]) - 0.2
            measurements.append(measurement)  
            


# In[24]:


X_train = np.array(images)
y_train = np.array(measurements)


# In[7]:



# # # Basic Network
# model = Sequential()
# model.add(Flatten(input_shape=(160,320,3)))
# model.add(Dense(1))
# model.compile(loss='mse', optimizer='adam')
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=7)
# model.save('model.h5')


# In[30]:


# # # Basic Network + Data Ppocessing
# model = Sequential()
# model.add(Lambda(lambda x:x/255.0 - 0.5, input_shape=(160,320,3)))
# model.add(Flatten())
# model.add(Dense(1))
# model.compile(loss='mse', optimizer='adam')
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)
# model.save('model2.h5')


# In[20]:


# # # # # LeNET

# class LeNet:
#     def build(height, width, channels, classes, weightsPath=None):   
        
#         model = Sequential() 
#         model.add(Lambda(lambda x:x/127.5-1.0, input_shape=(height,width,channels)))
#         #LeNET
#         model.add(Conv2D(filters=20,kernel_size=(5,5),  strides=(1, 1), padding="same"))
#         model.add(Activation("relu"))
#         model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

#         model.add(Conv2D(filters=50,kernel_size=(5,5),padding="same"))
#         model.add(Activation("relu"))
#         model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

#         model.add(Flatten())
#         model.add(Dense(500))
#         model.add(Activation("relu"))
#         model.add(Dense(classes))
#         model.add(Activation("softmax"))
#         model.summary()

#         return model


# In[21]:


# # # model 3 output
# model = LeNet.build(height=160, width = 320, channels=3, classes=1)
# model.compile(loss='mse', optimizer='adam')
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)
# model.save('model3.h5')


# In[28]:


augmented_images, augmented_measurements =[], []

for image, measurement in zip(images, measurements):
    #put rela iamges
    augmented_images.append(image)
    augmented_measurements.append(measurement)  
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)


# In[29]:


X_train_aug = np.array(augmented_images)
y_train_aug = np.array(augmented_measurements)


# In[30]:


IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 160, 320, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


# In[32]:




class NVIDIANet:
    def build(height, width, channels, classes):   
        
        model = Sequential()    
        model.add(Lambda(lambda x:x/127.5-1.0, input_shape=(height,width,channels)))  # normalize    
        model.add(Cropping2D(cropping=((70,25), (0,0)))) # cropped
        #conv
        model.add(Conv2D(filters=24, kernel_size=(5,5), strides=(2,2), input_shape = (height, width, channels)))
        model.add(Conv2D(filters=36, kernel_size=(5,5), strides=(2,2)))
        model.add(Conv2D(filters=48, kernel_size=(5,5), strides=(2,2)))
        model.add(Conv2D(filters=64, kernel_size=(3,3)))
        model.add(Conv2D(filters=64, kernel_size=(3,3)))      
        model.add(Dropout(0.5))
        #Dense
        model.add(Flatten())      
        model.add(Dense(100, activation ="elu"))
        model.add(Dense(50, activation = "elu"))
        model.add(Dense(10, activation = "elu"))  
        
        #output
        model.add(Dense(classes))
        
        #Summary
        model.summary()

        return model


# In[33]:


model = NVIDIANet.build(height=IMAGE_HEIGHT, width=IMAGE_WIDTH, channels=IMAGE_CHANNELS, classes=1)
model.compile(loss='mse', optimizer='adam')
model.fit(X_train_aug, y_train_aug, validation_split=0.2, shuffle=True, epochs=3)
model.save('model.h5')

