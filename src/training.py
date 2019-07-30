#importing libraries
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.layers import Activation, Convolution2D, Dropout, Conv2D
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras import layers
from keras.regularizers import l2
from models import mini_xception
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import cv2
import numpy as np
import os
import preprocessing as pre

# Getting required directories
# Model Directory
mod_dir = "/".join(os.getcwd().split("/")[0:-1] + ['model/'])
# Data Directory
data_dir = "/".join(os.getcwd().split("/")[0:-1] + ['data/'])
# Base Directory
base_dir = "/".join(os.getcwd().split("/")[0:-1])


# data loading and preprocessing
faces, emotions = pre.load_data(data_dir + "fer2013.csv")
faces = pre.preprocess_input(faces)
xtrain, xtest,ytrain,ytest = train_test_split(faces, emotions,test_size=0.2,shuffle=True)

# parameters
batch_size = 64
num_epochs = 100
input_shape = (48, 48, 1)
verbose = 1
num_classes = 7
patience = 30
l2_regularization=0.01

# data generator
data_generator = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)

# Build model
model = mini_xception(input_shape, num_classes)

# Model Compilation
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Defining callbacks
log_file_path = base_dir+"logs"+'training.log'
csv_logger = CSVLogger(log_file_path, append=False)
early_stop = EarlyStopping('val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience/4), verbose=1)
trained_models_path = mod_dir+'_mini_xception'
model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,save_best_only=True)
callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

# Model Fitting
model.fit_generator(data_generator.flow(xtrain, ytrain,batch_size),
                        steps_per_epoch=len(xtrain) / batch_size,
                        epochs=num_epochs, verbose=1, callbacks=callbacks,
                        validation_data=(xtest,ytest))

# Model Evaluation
train_score = model.evaluate(x_train, y_train, verbose=0)
print(train_score)
test_score = model.evaluate(x_test, y_test, verbose=0)
print(test_score)

# saving the final model
model.save("emotionrecognizer.h5")
