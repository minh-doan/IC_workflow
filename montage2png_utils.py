import glob
import math
import os
import os.path
import random
import warnings
import re
import shutil

import javabridge
import keras.utils.np_utils
import numpy
import skimage.exposure
import skimage.io
import skimage.measure
import skimage.morphology


def channel_regex(channels):
    return ".*" + "Ch(" + "|".join(str(channel) for channel in channels) + ")"

def class_weights(directory, data):
    """
    Compute the contribution of data from each class.

    :param directory: A directory containing class-labeled subdirectories containing .PNG images.
    :param data: A dictionary of class labels to directories containing .TIF files of that class. E.g.,
                     directory = {
                         "abnormal": "data/raw/abnormal",
                         "normal": "data/raw/normal"
                     }
    :return: A dictionary of class labels and contributions (as a decimal percentage), compatible with Keras.
    """
    counts = {}

    for label_index, label in enumerate(sorted(data.keys())):
        count = len(glob.glob("{}/{}/*.png".format(directory, label)))

        counts[label_index] = count

    total = max(sum(counts.values()), 1)

    for label_index, count in counts.items():
        counts[label_index] = count / total

    return counts

def parse(directory, data, channels, image_size):
    """
    Extracts single-channel .PNG images, cropped from .TIF files.

    Extracted images are saved to the following directory structure:
        directory/
            class_label_0/
                class_label_0_XX_YYYY_ZZ.png
                class_label_0_XX_YYYY_ZZ.png
                ...
            class_label_1/
                class_label_1_XX_YYYY_ZZ.png
                class_label_1_XX_YYYY_ZZ.png
                ...

    This directory structure can be processed by split to create training/validation and test sets.

    :param directory: The directory where extracted images are saved. The directory is assumed to be empty and will be
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
        
    for label, data_directory in data.items():
        if not os.path.exists("{}/{}".format(directory, label)):
            os.makedirs("{}/{}".format(directory, label))

        filenames = glob.glob("{}/*.tif".format(data_directory))
        
        filenames = [filename for filename in filenames if re.match(regex, os.path.basename(filename))]

        # When working with montage TIFF, how to group corresponding channels of the same object???
        for montage_id, filename in enumerate(sorted(filenames)):       
            _parse_montages(filename, label, montage_id, directory, channels, image_size)

    warnings.resetwarnings()    
    

def split(directory, labels, split, image_size):
    """
    Shuffle and split image data into training/validation and test sets.

    Generates four files for use with training:
        directory/test_x.npy
        directory/test_y.npy
        directory/training_x.npy
        directory/training_y.npy

    :param directory: A directory containing class-labeled subdirectories containing single-channel .PNG or .TIF images.
    :param data: A list of class labels.
    :param split: Percentage of data (as a decimal) assigned to the training/validation set.
    """
    #filenames = []

    labels = sorted(labels)

    for label in labels:
        filenames = []
        
        label_pngs = glob.glob(os.path.join(directory, label, "*.png"))

        # All files in this "directory" are processed_cropped files, they are ".png" anyway
        #label_tifs = glob.glob(os.path.join(directory, label, "*.tif"))
        
        #filenames = numpy.concatenate((filenames, label_pngs, label_tifs))
        filenames = numpy.concatenate((filenames, label_pngs))
                
        random.shuffle(filenames)
        
        # Convert ratio in "split" into number of objects:
        ss = numpy.array(numpy.cumsum(list(split.values())) * len(filenames), dtype = numpy.int64)
    
        for s in range(3):
            
            if s == 0:
                files_to_move = filenames[:ss[0]]
            else:
                files_to_move = filenames[ss[s-1]:ss[s]]
            
            subdir_name = os.path.join(directory, list(split.keys())[s], label)
            if not os.path.exists(subdir_name):
                os.makedirs(subdir_name)
            
            for f in files_to_move:
                
                # move file to current dir
                f_base = os.path.basename(f)
                shutil.move(f, os.path.join(subdir_name, f_base))
               
    for s in range(3):
        
        relocated_filenames = []
        for label in labels:
            # There might be nothing to concatenate, so this won't work:
            # relocated_filenames = numpy.concatenate((relocated_filenames, relocated_label_pngs))
            
            relocated_label_pngs = glob.glob(os.path.join(directory, list(split.keys())[s], label, "*.png"))
            relocated_filenames.append(relocated_label_pngs)
        
        if list(split.keys())[s] == "Training":
            # flatten the nested list of filenames
            training_filenames = numpy.array([item for sublist in relocated_filenames for item in sublist])
        
        if list(split.keys())[s] == "Validation":
            validation_filenames = numpy.array([item for sublist in relocated_filenames for item in sublist])
        
        if list(split.keys())[s] == "Testing":
            testing_filenames = numpy.array([item for sublist in relocated_filenames for item in sublist])            
        
    for name, filenames in [("training", training_filenames), ("validation", validation_filenames), ("testing", testing_filenames)]:
        x, y = _concatenate(filenames, labels, image_size)

        numpy.save(os.path.join(directory, "{}_x.npy".format(name)), x)

        numpy.save(os.path.join(directory, "{}_y.npy".format(name)), y)


def _concatenate(filenames, labels, image_size):
    collection = skimage.io.imread_collection(filenames)

    x = collection.concatenate().reshape((-1, image_size, image_size, 1))

    y = [labels.index(os.path.split(os.path.dirname(filename))[-1]) for filename in filenames]

    return x, keras.utils.np_utils.to_categorical(y)


def _parse_montages(filename, label, montage_id, directory, channels, image_size):
    montage = skimage.io.imread(filename)
    
    nrows = int(montage.shape[0] / image_size)

    ncols = int(montage.shape[1] / image_size)

    images = montage.reshape(nrows, image_size, ncols, image_size)

    images = images.swapaxes(1,2)

    images = images.reshape(-1, image_size, image_size)
    
    images = [image for image in images if numpy.std(image[:, int(image_size / 2)]) > 0]
    
    for image_id, image in enumerate(images):
        
        rescaled = skimage.exposure.rescale_intensity(
            image,
            out_range=numpy.uint16
        ).astype(numpy.uint16)
        
        skimage.io.imsave(
            "{}/{}/{}_{:02d}_{:05d}.png".format(
                directory,
                label,
                label,
                montage_id,
                image_id
            ),
            rescaled,
            plugin="imageio"
        )  