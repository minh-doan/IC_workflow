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

        # When working with small TIFF, sorted(filenames) groups corresponding channels of the same object
        for file_id, filename in enumerate(sorted(filenames)): 
            _parse_tif(filename, label, file_id, directory, channels, image_size)

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


def _parse_tif(filename, label, file_id, directory, channels, image_size):

    image = skimage.io.imread(filename)

    cropped = _crop(image, image_size)
        
    rescaled = skimage.exposure.rescale_intensity(
        cropped,
        out_range=numpy.uint8
    ).astype(numpy.uint8)
        
    skimage.io.imsave(
        "{}/{}/{}_{:02d}.png".format(
            directory,
            label,
            label,
            file_id
        ),
        rescaled,
        plugin="imageio"
    )
