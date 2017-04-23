
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


input_directory = "/data1/Minh/Leukemia/DeepLearning/Allchannels_digested_except113_190_207_209/"

output_directory = "/home/jccaicedo/Leukemia_DeepLearning/BFDFDAPI_test209_190pres_heavyweights"


data = {
    "normal": '/data1/Minh/IFC/DeepLearning/cif/normal/',
    
  "leukemic": '/data2/Minh/IFC/DeepLearning/cif/abnormal/'
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

# Use this function to rescale signal intensities across images
def min_max_norm(x, minimum=None, maximum=None):
    channels = x.shape[-1]
    if minimum is None and maximum is None:
        minimum = []
        maximum = []
        for channel in range(channels):
            minimum.append( x[..., channel].min() )
            maximum.append( x[..., channel].max() )
    result = numpy.zeros_like(x)
    for ch in range(channels):
        result[..., ch] = 100.0*( (numpy.ndarray.astype(x[..., ch], numpy.float32) - minimum[ch])/(maximum[ch] - minimum[ch]) )
    return (result, minimum, maximum)

### DATA QUEUEING

def training_data_generator(input_x, input_y, batch_size):
    num_examples, num_labels = input_y.shape
    label_indices = []
    for i in range(num_labels):
        indices = [j for j in range(num_examples) if input_y[j,i] > 0]
        label_indices.append(indices)
        print("Label",i,":",len(indices),"examples")
    samples_per_label = int(batch_size / num_labels)

    def generator():
        while True:
            x_samples = []
            y_samples = []
            for i in range(num_labels):
                random.shuffle(label_indices[i])
                indices = label_indices[i][0:samples_per_label]
                x_samples.append( input_x[indices, ...] )
                y_samples.append( input_y[indices, ...] )
            x_samples = numpy.concatenate( x_samples )
            y_samples = numpy.concatenate( y_samples )
            batch_indices = numpy.arange(x_samples.shape[0])
            numpy.random.shuffle(batch_indices)
            x_samples = x_samples[batch_indices, ...]
            y_samples = y_samples[batch_indices, ...]
            yield (x_samples, y_samples)
    return generator()


def prediction_data_generator(input_x, input_y, batch_size):
    num_examples, num_labels = input_y.shape
    steps = int(num_examples / batch_size)
    def generator():
        i = 0
        while True:
            start = i*batch_size
            end = (i+1)*batch_size
            x_sample = input_x[start:end, ...]
            y_sample = input_y[start:end, ...]
            yield (x_sample, y_sample)
            i = i + 1 if i < steps else 0
    print("Prediction steps:",steps)        
    return generator(), steps

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

print("Loading training data")

training_x = numpy.load(os.path.join(input_directory, "augmented_training_x.npy"))

training_y = numpy.load(os.path.join(input_directory, "augmented_training_y.npy"))

# Use this function to normalize signal intensities across images
training_x, pix_min, pix_max = min_max_norm(training_x)

training_generator = training_data_generator(training_x, training_y, 32) 

print(training_x.shape, training_y.shape)


# In[9]:

print("Loading validation data")

validation_x = numpy.load(os.path.join(input_directory, "validation_x.npy"))

validation_y = numpy.load(os.path.join(input_directory, "validation_y.npy"))

# Use this function to normalize signal intensities across images
validation_x, _, _ = min_max_norm(validation_x, pix_min, pix_max)

validation_generator, validation_steps = prediction_data_generator(validation_x, validation_y, 32)

print(validation_x.shape)

# In[12]:

print("Loading test data")

testing_x = numpy.load(os.path.join(input_directory, "testing_x.npy"))

testing_y = numpy.load(os.path.join(input_directory, "testing_y.npy"))

# Use this function to normalize signal intensities across images
testing_x, _, _ = min_max_norm(testing_x, pix_min, pix_max)

testing_generator, testing_steps = prediction_data_generator(testing_x, testing_y, 32)

print(testing_x.shape)

# # Construct convolutional neural network:

# In[15]:

shape = (training_x.shape[1], training_x.shape[2], training_x.shape[3])

x = keras.layers.Input(shape)


# In[16]:

options = {"activation": None, "kernel_size": (3, 3), "padding": "same"}

# Block 1:

y = keras.layers.Conv2D(32, **options)(x)
y = keras.layers.normalization.BatchNormalization()(y)
y = keras.layers.Activation("relu")(y)

y = keras.layers.Conv2D(32, **options)(y)
y = keras.layers.Activation("relu")(y)
y = keras.layers.normalization.BatchNormalization()(y)

y = keras.layers.MaxPooling2D(pool_size=2, strides=None, padding='same')(y)

# Block 2:
y = keras.layers.Conv2D(64, **options)(y)
y = keras.layers.Activation("relu")(y)
y = keras.layers.normalization.BatchNormalization()(y)

y = keras.layers.Conv2D(64, **options)(y)
y = keras.layers.Activation("relu")(y)
y = keras.layers.normalization.BatchNormalization()(y)

y = keras.layers.MaxPooling2D(pool_size=2, strides=None, padding='same')(y)

# Block 3:
y = keras.layers.Conv2D(128, **options)(y)
y = keras.layers.Activation("relu")(y)
y = keras.layers.normalization.BatchNormalization()(y)

y = keras.layers.Conv2D(128, **options)(y)
y = keras.layers.Activation("relu")(y)
y = keras.layers.normalization.BatchNormalization()(y)

y = keras.layers.MaxPooling2D(pool_size=2, strides=None, padding='same')(y)

# Block 4:
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

optimizer = keras.optimizers.Adam(0.0001)

model.compile(
    loss=loss, 
    metrics=[
        "accuracy"
    ],
    optimizer=optimizer
)

# # Train the network

# In[ ]:

# New output output_directory:
if not os.path.exists(output_directory):
    os.makedirs(output_directory)


# In[ ]:

csv_logger = keras.callbacks.CSVLogger(os.path.join(output_directory, 'training.csv') )
early_stopping = keras.callbacks.EarlyStopping(patience=64)

# checkpoint
filepath = os.path.join(output_directory, "weights.best.hdf5")
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
        epochs=5,
        class_weight = class_weights,
        generator=training_generator,
        max_q_size=4096,
        steps_per_epoch=2000,
        validation_data=validation_generator,
        validation_steps=validation_steps
    )


# Evaluate testing set

# In[ ]:

model.evaluate_generator(
    generator=testing_generator, 
    steps=testing_steps
)


session.close()


# In[ ]:



