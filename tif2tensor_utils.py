import glob
import math
import os
import os.path
import random
import warnings
import re
import shutil

import keras.utils.np_utils
import numpy
import skimage.exposure
import skimage.io
import skimage.measure
import skimage.morphology


def channel_regex(channels):
    return ".*" + "Ch(" + "|".join(str(channel) for channel in channels) + ")"

def parse(directory, data, channels, image_size, split):
    """

    :param directory: The directory where temporary processed files are saved. The directory is assumed to be empty and will be
                      created if it does not exist.
    :param data: A dictionary of class labels to directories containing .CIF files of that class. E.g.,
                     directory = {
                         "abnormal": "data/raw/abnormal",
                         "normal": "data/raw/normal"
                     }
    :param channels: An array of channel indices (0 indexed). Only these channels are extracted. Unlisted channels are
                     ignored.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    warnings.filterwarnings("ignore")
    
    regex = channel_regex(channels)

    nested_filenames = []
    
    for label, data_directory in sorted(data.items()):

        filenames = glob.glob("{}/*.tif".format(data_directory))
        
        filenames = [filename for filename in filenames if re.match(regex, os.path.basename(filename))]
        
        nested_filenames.append(sorted(filenames))
      
    multichannel_tensors = []
    onehot_labels = []
    for i in range(len(nested_filenames)): # each list in the nest is data of 1 label, each list contains multiple channels
        
        single_channel_tensors = []
        
        for j in range(len(channels)):
            
            cropped_images = [_crop(skimage.io.imread(filename),image_size) for filename in nested_filenames[i][j::len(channels)] ] # tensor rank 3
            
            single_channel_tensors.append(numpy.array(cropped_images)) # nested list of tensor rank 3
            
        multichannel_tensor = numpy.array(single_channel_tensors) # tensor rank 4, images of one label
        multichannel_tensors.append(multichannel_tensor) # nested list of tensor rank 4, images of all the labels, pay attention to the plural "tensorsssss"
        
        onehot_label = numpy.zeros((multichannel_tensor.shape[1],len(nested_filenames)))
        onehot_label[:multichannel_tensor.shape[0],i] = 1
        onehot_labels.append(onehot_label)
                 
    # Final tensor and labels:  
    T = numpy.concatenate((multichannel_tensors), axis = 1).swapaxes(0, 1).swapaxes(1, 2).swapaxes(2, 3) # because T.shape was (channels, batch, width, height)
    L = numpy.concatenate((onehot_labels))
    print('All images are saved inside this tensor rank 4, "Tensor", shape: ' + str(T.shape))
    print('All labels are encoded in this one-hot label tensor rank 2, "Labels" ,shape: ' + str(L.shape))
    
    numpy.save(os.path.join(directory, "{}.npy".format('Tensor')), T)

    numpy.save(os.path.join(directory, "{}.npy".format('Labels')), L)
    
    warnings.resetwarnings()

    #---------------- Splitting ----------------#
    
    training_images = []
    validation_images = []
    testing_images = []
    training_label = []
    validation_label = []
    testing_label = []
       
    for t in range(len(multichannel_tensors)):
        tensor = multichannel_tensors[t].swapaxes(0, 1).swapaxes(1, 2).swapaxes(2, 3) # multichannel_tensors[t].shape was (channels, batch, width, height)
        
        # Convert ratio in "split" into number of objects:
        ss = numpy.array(numpy.cumsum(list(split.values())) * tensor.shape[0], dtype = numpy.int64) 
        random.shuffle(tensor) 
           
        training_images.append(tensor[:ss[0]])
        validation_images.append(tensor[ss[0]:ss[1]])
        testing_images.append(tensor[ss[1]:ss[2]])
        
        onehot_l = numpy.zeros((tensor.shape[0],len(multichannel_tensors)))
        onehot_l[:tensor.shape[0],t] = 1
        
        training_label.append(onehot_l[:ss[0]])
        validation_label.append(onehot_l[ss[0]:ss[1]])
        testing_label.append(onehot_l[ss[1]:ss[2]])          
                    
    numpy.save(os.path.join(directory, "training_x.npy"), numpy.concatenate((training_images)) ) # flatten nested list
    numpy.save(os.path.join(directory, "training_y.npy"), numpy.concatenate((training_label)) ) 
    print('Training tensor "training_x" was saved, shape: ' + str(numpy.concatenate((training_images)).shape) )

    numpy.save(os.path.join(directory, "validation_x.npy"), numpy.concatenate((validation_images)) )
    numpy.save(os.path.join(directory, "validation_y.npy"), numpy.concatenate((validation_label)) )
    print('Validation tensor "validation_x" was saved, shape: ' + str(numpy.concatenate((validation_images)).shape) )
          
    numpy.save(os.path.join(directory, "testing_x.npy"), numpy.concatenate((testing_images)) )
    numpy.save(os.path.join(directory, "testing_y.npy"), numpy.concatenate((testing_label)) )
    print('Testing tensor "testing_x" was saved, shape: ' + str(numpy.concatenate((testing_images)).shape) )
        
    #---------------- Class weight ----------------#
    
    counts = {}

    print('Number of objects in each class:')
    for label_index, label in enumerate(sorted(data.keys())):
        
        count = multichannel_tensors[label_index].shape[1] # multichannel_tensors[].shape was (channels, batch, width, height)
        
        print(label_index, label, count)

        counts[label_index] = count

    total = max(sum(counts.values()), 1)

    for label_index, count in counts.items():
        counts[label_index] = count / total    
    
    print('Class weight : ',counts)
    return counts


def _crop(image, image_size):
    
    bigger = max(image.shape[0], image.shape[1], image_size)

    pad_x = float(bigger - image.shape[0])
    pad_y = float(bigger - image.shape[1])

    pad_width_x = (int(math.floor(pad_x / 2)), int(math.ceil(pad_x / 2)))
    pad_width_y = (int(math.floor(pad_y / 2)), int(math.ceil(pad_y / 2)))
    # Sampling the background, avoid the corners which may have contaminated artifacts
    sample = image[int(image.shape[0]/2)-4:int(image.shape[0]/2)+4, 3:9]

    std = numpy.std(sample)

    mean = numpy.mean(sample)

    def normal(vector, pad_width, iaxis, kwargs):
        vector[:pad_width[0]] = numpy.random.normal(mean, std, vector[:pad_width[0]].shape)
        vector[-pad_width[1]:] = numpy.random.normal(mean, std, vector[-pad_width[1]:].shape)
        return vector

    if (image_size > image.shape[0]) & (image_size > image.shape[1]):
        return numpy.pad(image, (pad_width_x, pad_width_y), normal)
    else:
        if bigger > image.shape[1]:
            temp_image = numpy.pad(image, (pad_width_y), normal)
        else:
            if bigger > image.shape[0]:
                temp_image = numpy.pad(image, (pad_width_x), normal)
            else:
                temp_image = image
                
        center_x = int(temp_image.shape[0] / 2.0)

        center_y = int(temp_image.shape[1] / 2.0)
        
        radius = int(image_size/2)
        
        cropped = temp_image[center_x - radius:center_x + radius, center_y - radius:center_y + radius]
        
        assert cropped.shape == (image_size, image_size), cropped.shape
                
        return cropped