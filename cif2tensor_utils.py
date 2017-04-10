#----- GPU cluster didn't have python-bioformats after reset ----#
#import pip

#def install(package):
#   pip.main(['install', package, '--user'])
#import os
#import sys
#install('python-bioformats')


import glob
import math
import os
import os.path
import random
import warnings

import bioformats
import bioformats.formatreader
import javabridge
import keras.utils.np_utils
import numpy
import skimage.exposure
import skimage.io
import skimage.measure
import skimage.morphology
javabridge.start_vm(class_path=bioformats.JARS, max_heap_size='32G')

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
       
    multichannel_tensors = []
    onehot_labels = []
    index = 0
    
    for label, data_directory in sorted(data.items()):

        filenames = glob.glob("{}/*.cif".format(data_directory))
        print('List of .cif files in this folder: ',filenames)
        
        temp_tensor = []
        for filename in filenames:
            single_channel_tensors = []
            print('Now parsing: ',filename)            
            reader = bioformats.formatreader.get_image_reader("tmp", path=filename)

            image_count = javabridge.call(reader.metadata, "getImageCount", "()I")
            channel_count = javabridge.call(reader.metadata, "getChannelCount", "(I)I", 0)

            for channel in channels:

                images = [reader.read(c=channel, series=image) for image in range(image_count)[::2]]

                cropped_images = numpy.expand_dims([_crop(image, image_size) for image in images], axis =3) # tensor rank 3

                single_channel_tensors.append(cropped_images) # nested list of tensor rank 3 (film strips)

            multichannel_tensor = numpy.concatenate((single_channel_tensors), axis = 3) # tensor rank 4, images of one .cif
            
            # Done digesting a .cif file, store it:
            temp_tensor.append(multichannel_tensor) # nested list of tensor rank 4, contain images of all the .cif files of this label

        temp_tensor_2 = numpy.concatenate((temp_tensor))
        
        onehot_label = numpy.zeros((temp_tensor_2.shape[0],len(data.keys() ) ) )
        onehot_label[:,index] = 1
        
        index += 1
        
        multichannel_tensors.append(temp_tensor_2) # nested list of tensor rank 4, contain images of all the .cif files of ALL labels
        onehot_labels.append(onehot_label)
        
    # Final tensor and labels:  
    T = numpy.concatenate((multichannel_tensors))
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
        tensor = multichannel_tensors[t]
        
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
        
        count = multichannel_tensors[label_index].shape[0]
        
        print(label_index, label, count)

        counts[label_index] = count

    total = max(sum(counts.values()), 1)

    for label_index, count in counts.items():
        counts[label_index] = count / total    
    
    numpy.save(os.path.join(directory, "class_weights.npy"), counts)
    print('Class weight(s) : ',counts)
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