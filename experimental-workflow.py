
# coding: utf-8

# # Available digest modules:
# cif2png_utils, tif2png_utils, montage2png_utils, tif2tensor_utils, cif2tensor_utils

# In[1]:

import cif2tensor_utils


# In[2]:

import keras
import numpy
import matplotlib.pyplot
import os.path
import pandas
import seaborn
import sklearn.metrics
import keras.applications
import keras.preprocessing.image
import tensorflow
import random


# # User's settings

# In[3]:

directory = "/data1/Minh/IFC/DeepLearning/BFDFDAPI_digested_except113_190_207_209"

data = {
    "normal": '/data1/Minh/IFC/DeepLearning/cif/normal/',
    
  "leukemic": '/data1/Minh/IFC/DeepLearning/cif/abnormal/'
}

# Warning: Neural networks often require a combination of 1 or 3 or 4 channels.
# Users should specify ALL desired channels here. In downstream modules, users can duplicate channels if needed.
channels = [0,5,6]
number_of_channels = 2

image_size = 28

split = {
    "Training" : 0.8,
    "Validation" : 0.1,
    "Testing" : 0.1
}

classes = len(data.keys())


# Heavy weights
class_weights = {0: 1.0300000712336965, 1: 34.333254184969725}



# # Load data and labels:

# In[5]:

# Use this function to normalize signal intensities across images
def min_max_norm(x, minimum=None, maximum=None):
    if minimum is None:
        minimum = x.min()
    if maximum is None:
        maximum = x.max()
    result = 100.0*( (numpy.ndarray.astype(x, numpy.float32) - minimum)/(maximum - minimum) )
    return (result, minimum, maximum)

### DATA QUEUEING

def data_generator(input_x, input_y, batch_size):
    def generator():
        while True:
            indices = sorted( random.sample(range(input_x.shape[0]), batch_size) )
            x_sample = input_x[indices, ...]
            y_sample = input_y[indices, ...]
            yield (x_sample, y_sample)
    return generator()


'''
#TODO: Prefill the queue
#TODO: Balance data
#TODO: Add augmentations
def data_generator(input_x, input_y, batch_size, session, scope="training"):
    # Prepare the batch queue
    x_shape = list(input_x.shape)
    x_shape[0] = batch_size
    y_shape = list(input_y.shape)
    y_shape[0] = batch_size
    x_ph = tensorflow.placeholder(tensorflow.float32, shape=x_shape, name="x_ph")
    y_ph = tensorflow.placeholder(tensorflow.float32, shape=y_shape, name="y_ph")
    x_batch, y_batch = tensorflow.train.shuffle_batch(
        [x_ph, y_ph],
        batch_size=batch_size,
        capacity=batch_size,
        min_after_dequeue=batch_size
    )
    # Push data to queue
    def generator():
        while True:
            indices = sorted( random.sample(range(input_x.shape[0]), batch_size) )
            x_sample = input_x[indices, ...]
            y_sample = input_y[indices, ...]
            x, y = session.run([x_batch, y_batch], feed_dict={x_ph:x_sample, y_ph:y_sample})
            #print("Really?",x.shape)
            yield (x_sample, y_sample)
    # Return generator
    return generator()
'''


# In[6]:

training_x = numpy.load(os.path.join(directory, "training_x.npy"))

training_y = numpy.load(os.path.join(directory, "training_y.npy"))

print(training_x.shape)

# Use this function to normalize signal intensities across images
training_x, pix_min, pix_max = min_max_norm(training_x)

training_generator = data_generator(training_x, training_y, 32) 


# In[9]:

validation_x = numpy.load(os.path.join(directory, "validation_x.npy"))

validation_y = numpy.load(os.path.join(directory, "validation_y.npy"))

# Use this function to normalize signal intensities across images
validation_x, pix_min, pix_max = min_max_norm(validation_x, pix_min, pix_max)

validation_generator = data_generator(validation_x, validation_y, 32)

# In[12]:

testing_x = numpy.load(os.path.join(directory, "testing_x.npy"))

testing_y = numpy.load(os.path.join(directory, "testing_y.npy"))

# Use this function to normalize signal intensities across images
testing_x, pix_min, pix_max = min_max_norm(testing_x, pix_min, pix_max)

test_generator = data_generator(testing_x, testing_y, 32)
 

# # Construct convolutional neural network:

# In[15]:

shape = (training_x.shape[1], training_x.shape[2], training_x.shape[3])

x = keras.layers.Input(shape)


# In[16]:

options = {"activation": "relu", "kernel_size": (3, 3), "padding": "same"}

# Block 1:

y = keras.layers.Conv2D(32, **options)(x)
y = keras.layers.Conv2D(32, **options)(y)

y = keras.layers.MaxPooling2D(pool_size=2, strides=None, padding='same')(y)

# Block 2:
y = keras.layers.Conv2D(64, **options)(y)
y = keras.layers.Conv2D(64, **options)(y)

y = keras.layers.MaxPooling2D(pool_size=2, strides=None, padding='same')(y)

# Block 3:
y = keras.layers.Conv2D(128, **options)(y)
y = keras.layers.Conv2D(128, **options)(y)

y = keras.layers.MaxPooling2D(pool_size=2, strides=None, padding='same')(y)

# Block 4:
# y = keras.layers.Conv2D(256, **options)(y)
# y = keras.layers.Conv2D(256, **options)(y)

# y = keras.layers.MaxPooling2D(pool_size=2, strides=None, padding='same')(y)

# Block 5:
y = keras.layers.Flatten()(y)

intermediate_layer = keras.layers.Dense(1024, activation="relu")(y) # This intermediate_layer will be used for embeddings

y = keras.layers.Dropout(0.5)(intermediate_layer)

y = keras.layers.Dense(classes)(y)

y = keras.layers.Activation("softmax")(y)


# In[17]:

model = keras.models.Model(x, y)

# In[18]:

model.summary()


# In[19]:

loss = keras.losses.categorical_crossentropy

optimizer = keras.optimizers.Adam(0.00001)

model.compile(
    loss=loss, 
    metrics=[
        "accuracy"
    ],
    optimizer=optimizer
)

# # Train the network

# In[ ]:

# New output directory:
directory = "/home/jccaicedo/Leukemia_DeepLearning/BFDFDAPI_test209_190pres_heavyweights"
if not os.path.exists(directory):
    os.makedirs(directory)


# In[ ]:

csv_logger = keras.callbacks.CSVLogger(os.path.join(directory, 'training.csv') )

early_stopping = keras.callbacks.EarlyStopping(patience=64)

# checkpoint
filepath = os.path.join(directory, "weights.best.hdf5")
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_categorical_accuracy', verbose=1, save_best_only=True, mode='max')


# In[ ]:

configuration = tensorflow.ConfigProto()

configuration.gpu_options.allow_growth = True

session = tensorflow.Session(config=configuration)

keras.backend.set_session(session)


# In[ ]:

with tensorflow.device("/gpu:0"):
    model.fit_generator(
        callbacks=[
            #checkpoint,
            csv_logger
        ],
        epochs=15,
        class_weight = class_weights,
        generator=training_generator,
        max_q_size=256,
        steps_per_epoch=2000,
        validation_data=validation_generator,
        validation_steps=2000
    )


# Evaluate testing set

# In[ ]:

model.evaluate_generator(
    generator=test_generator, 
    steps=256
)


# In[ ]:

testing_x = numpy.load("/home/minh-doan/Leukemia_DeepLearning/BFDFDAPI_digested_test209pres/testing_x.npy")

testing_y = numpy.load("/home/minh-doan/Leukemia_DeepLearning/BFDFDAPI_digested_test209pres/testing_y.npy")

# If using networks (VGG19) that needs 3 channels RGB not single-channel grayscale:
# Examples:
# testing_xx = numpy.concatenate((testing_x,testing_x,testing_x), axis=3)
# testing_xx = numpy.concatenate((testing_x,numpy.expand_dims(testing_x[:,:,:,0], axis = 3)), axis=3)
# testing_xx = numpy.concatenate((testing_x,testing_x), axis=3)


# In[ ]:

# Use this function to normalize signal intensities across images
testing_x, pix_min, pix_max = min_max_norm(testing_x, pix_min, pix_max)


# In[ ]:

test_generator = keras.preprocessing.image.ImageDataGenerator() #rotation_range = 180, horizontal_flip = True, vertical_flip = True)

test_generator = test_generator.flow(
    x = testing_x, # or testing_xx
    y = testing_y,
    batch_size=32
)

# If using PNG from folders:

# test_generator = test_generator.flow_from_directory(
#     batch_size=1,
#     color_mode="rgb",
#     directory="/home/minh-doan/Cell_cycle/temp_processed/Testing/"
# )


# In[ ]:

model.evaluate_generator(
    generator=test_generator, 
    steps=256
)


# In[ ]:

testing_x = numpy.load("/home/minh-doan/Leukemia_DeepLearning/BFDFDAPI_digested_test190pres/testing_x.npy")

testing_y = numpy.load("/home/minh-doan/Leukemia_DeepLearning/BFDFDAPI_digested_test190pres/testing_y.npy")

# If using networks (VGG19) that needs 3 channels RGB not single-channel grayscale:
# Examples:
# testing_xx = numpy.concatenate((testing_x,testing_x,testing_x), axis=3)
# testing_xx = numpy.concatenate((testing_x,numpy.expand_dims(testing_x[:,:,:,0], axis = 3)), axis=3)
# testing_xx = numpy.concatenate((testing_x,testing_x), axis=3)


# In[ ]:

# Use this function to normalize signal intensities across images
testing_x, pix_min, pix_max = min_max_norm(testing_x, pix_min, pix_max)


# In[ ]:

test_generator = keras.preprocessing.image.ImageDataGenerator() #rotation_range = 180, horizontal_flip = True, vertical_flip = True)

test_generator = test_generator.flow(
    x = testing_x, # or testing_xx
    y = testing_y,
    batch_size=32
)

# If using PNG from folders:

# test_generator = test_generator.flow_from_directory(
#     batch_size=1,
#     color_mode="rgb",
#     directory="/home/minh-doan/Cell_cycle/temp_processed/Testing/"
# )


# In[ ]:

model.evaluate_generator(
    generator=test_generator, 
    steps=256
)


# In[ ]:

session.close()


# In[ ]:



