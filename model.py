
# coding: utf-8

# In[1]:

######## Loading training data and defining the generator ########

#Loading training data
import csv
import cv2
import numpy as np
import sklearn

#Loading sample training data

lines = []
with open('sample_training_data/data/driving_log.csv') as csvfile:
    next(csvfile) #skipping header row
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
#Loading additional recovery driving data - targeted for the curve after the dirt (after the bridge)       
with open('my_training_data/recovery_driving_targeted/driving_log_MODIFIED_lower_steering_angles.csv') as csvfile:
    #next(csvfile) #skipping header row
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
#Loading additional dirt recovery data (dirt after the bridge)       
with open('my_training_data/recovery_driving_dirt/driving_log_MODIFIED_lower_steering_angles.csv') as csvfile:
    #next(csvfile) #skipping header row
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
#Loading additional bridge recovery data (vehicle had a tendency to veer off to the left before the bridge)       
with open('my_training_data/recovery_driving_bridge/driving_log_MODIFIED_even_lower_steering_angles.csv') as csvfile:
    #next(csvfile) #skipping header row
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                
                image_center = cv2.imread('sample_training_data/data/IMG/' + batch_sample[0].split('/')[-1])
                image_left = cv2.imread('sample_training_data/data/IMG/' + batch_sample[1].split('/')[-1])
                image_right = cv2.imread('sample_training_data/data/IMG/' + batch_sample[2].split('/')[-1]) 
                images.append(image_center)
                images.append(image_left)
                images.append(image_right)
    
                steering_correction = 0.245
        
                measurement_center = float(batch_sample[3])
                measurement_left = measurement_center + steering_correction
                measurement_right = measurement_center - steering_correction
                measurements.append(measurement_center)
                measurements.append(measurement_left)
                measurements.append(measurement_right)
                
                #Flipping images
                #images.append(cv2.flip(image_center,1))
                #images.append(cv2.flip(image_left,1))
                #images.append(cv2.flip(image_right,1))
                #measurements.append(measurement_center*-1.0)
                #measurements.append(measurement_left*-1.0)
                #measurements.append(measurement_right*-1.0)

            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)
            
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

print("Data loading generator defined!")

#print("Shape of X_train:", X_train.shape)
#print("Shape of y_train:", y_train.shape)


# In[2]:

######## Definition of model archiecture & training ########

# Importing all necessary Keras modules
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from keras.regularizers import l2
from keras.layers.advanced_activations import ELU

#define a lambda function for resizing images:
def myLambda(x):
    from keras.backend import tf as ktf
    return ktf.image.resize_images(x, (30, 60))

# my model architecture
model = Sequential()
model.add(Lambda(myLambda, input_shape=(160, 320, 3))) # resizing images
model.add(Cropping2D(cropping=((9,3), (0,0)))) #cropping images
model.add(Lambda(lambda x: (x / 255.0) - 0.5)) # normalizing data
model.add(Convolution2D(32, 5, 5))
model.add(MaxPooling2D((2, 2)))
#model.add(Dropout(0.75))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(MaxPooling2D((2, 2)))
#model.add(Dropout(0.75))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(1))

#Compile and train the model
model.compile(loss='mse', optimizer ='adam')
model.fit_generator(train_generator,
                    samples_per_epoch = (len(train_samples)//32)*32+32,
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples),
                    nb_epoch=5)

model.save('model_v05.h5')

print("Model trained and saved")


# In[ ]:



