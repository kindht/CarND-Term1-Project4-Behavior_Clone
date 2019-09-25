import csv
import numpy as np

import matplotlib.pyplot as plt 
import matplotlib.image as mpimg

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
#from keras.preprocessing.image import ImageDataGenerator

import math

# Define data path 
LOG_PATH = '../my-data/driving_log.csv'   
IMG_PATH = '../my-data/IMG/'


############################
#### Step 0.  Load Data ####
############################
samples = []
with open(LOG_PATH) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
#samples = samples[1:]

######################################################
#### Step 1. Generate Samples (Pre-process Data)  ####
######################################################
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
 
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # loop forever 
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset: offset + batch_size]

            images = []  # image samples
            angles = []  # steering angles
            for cur_sample in batch_samples:
                path_center = IMG_PATH + cur_sample[0].split('/')[-1]
                path_left   = IMG_PATH + cur_sample[1].split('/')[-1]
                path_right  = IMG_PATH + cur_sample[2].split('/')[-1]

                img_center = mpimg.imread(path_center)
                img_left   = mpimg.imread(path_left)
                img_right  = mpimg.imread(path_right)

                # create adjusted steering measurements for the side camera images
                correction = 0.2  # parameter to tune
                angle_center = float(cur_sample[3])
                angle_left   = angle_center + correction  
                angle_right  = angle_center - correction

                # flipped image samples
                image_flipped = np.fliplr(img_center)
                angle_flipped = -1.0 *angle_center

                images.extend([img_center, img_left, img_right, image_flipped])
                angles.extend([angle_center, angle_left, angle_right, angle_flipped])

                X_train = np.array(images)
                y_train = np.array(angles)
                yield shuffle(X_train, y_train)

# Set batch size
batch_size = 2048   # 32 for small data
drop_rate = 0.5
EPOCHS = 15

# Generate train /validation samples using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

##################################
#### Step 2. Build the Model  ####
##################################
img_shape = (160,320,3)   # height, width, color

model = Sequential()  
model.add(Cropping2D(cropping=((50,20), (0,0)),  input_shape = img_shape)) #  trimmed image
model.add(Lambda(lambda x: (x / 255.0 - 0.5))) # normalize data and mean centered around zero

def build_Basic_Model(model):
    model.add(Flatten())
    model.add(Dense(1))
    return model

# Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format='channels_last')
def build_LeNet_Model(model):
    model.add(Conv2D(6,  kernel_size=5, activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(16, kernel_size=5, activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84,  activation='relu'))
    model.add(Dense(1))   
    return model

def build_PilotNet_Model(model):
    model.add(Conv2D(24, kernel_size=5, strides=2, activation='relu'))
    model.add(Conv2D(36, kernel_size=5, strides=2, activation='relu'))
    model.add(Conv2D(48, kernel_size=5, strides=2, activation='relu'))
    model.add(Conv2D(64, kernel_size=3, strides=1, activation='relu'))
    model.add(Conv2D(64, kernel_size=3, strides=1, activation='relu'))
    model.add(Flatten())

    model.add(Dropout(drop_rate)) # add this layer when overfitting
    model.add(Dense(100, activation='relu'))  # orig=1164 - when dataset is very large
    model.add(Dropout(drop_rate))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(drop_rate))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(1))
    return model

#model = build_Basic_Model(model)
#model = build_LeNet_Model(model)
model = build_PilotNet_Model(model)

#model = load_model('model.h5')

#####################################################
####  Step 3. Train, Validate and Save the Model ####
#####################################################
model.compile(loss='mse', optimizer='adam')
history_obj = model.fit_generator(train_generator, 
                    steps_per_epoch=math.ceil(len(train_samples)/batch_size), 
                    validation_data=validation_generator, 
                    validation_steps=math.ceil(len(validation_samples)/batch_size), 
                    epochs=EPOCHS, verbose=1)
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)

model.save('model.h5')

##################################
####  Step 4. Visualize Loss  ####
##################################
# print the keys contained in the history object
print(history_obj.history.keys())

# plot the training and validation loss for each epoch
plt.plot(history_obj.history['loss'])
plt.plot(history_obj.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('history_plot')
