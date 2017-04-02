Imaging flow cytometry
======================

Imaging flow cytometry (IFC) combines flow cytometry and single-cell imaging. An imaging flow cytometer produces images that correspond to discrete cellular objects.

The imaging-flow-cytometry notebook demonstrates a simple and reproducible workflow for classifying phenotypes from images produced by an imaging flow cytometer. The notebook demonstrates the process for reading the CIF files produced by imaging flow cytometers, pre-processing, re-training the classifier, and making predictions. In addition, the notebook includes methods for assessing the accuracy of the trained classifier.

In addition to the example workflow and corresponding data, we’ve also provided weights pre-trained on the example data with the LeNet-5-like convolutional neural network implemented in the imaging-flow-cytometry notebook. We encourage researchers use the pre-trained weights to avoid the tedious process of re-training the convolutional layers of the network when training their classifiers.
