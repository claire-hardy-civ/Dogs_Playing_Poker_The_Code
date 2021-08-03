from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
import numpy as np
import numpy as np
import os
import random
import sklearn
from matplotlib import pyplot as plt
import tensorflow
from tensorflow.keras import losses, layers, models, metrics, Model
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn import svm
from sklearn import metrics
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
#import torch
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
##
## Define a CNN
##

rotations_num = 4
augment_num = 10

saved_name = 'selfsupervised'
cnn_name = 'mycnn'

selfsupervised_epochs = 80
selfsupervised_batch_size = 128
supervised_epochs = 50
supervised_batch_size = 128
supervised_trainval_ratio = 1. / 6 #1/6 is implied in CIFAR-10

feature_layer_trained = 'conv2_block3_out'
feature_layer = 'conv2_block3_out'
first_resnet_layer = 'conv2_block1_out'
second_resnet_layer = 'conv2_block2_out'
feature_layer_cnn = 'out_layer'






##
## Split Data: Function to split data into known and unknown
##
## Params:
## 
##    x - numpy array of x's (images)
##    y - numpy array of 1-hot y's (labels)
##    toRemove - list of the class numbers to remove ex: [1,3]
##
## Returns:
##
##     x_known - numpy array of x's of the known classes
##     x_unknown - numpy array of x's of the unknown classes
##     y_known - numpy array of 1-hot y's of the known classes (column of the unknown classes have been removed...could be handled more elegantly)
##     y_unknown - numpy array of 1-hot y's of the unknown classes (column of the unknown classes has been removed...should all be zeros)
##
def split_data(x, y, toRemove):
    
    ##
    ## Y is one-hot, so need to find the number of the class (that is the format toRemove has)
    ##
    y_comp =np.argmax(y, axis = 1)                ## Turn 1-hot into numeric classes
    
    ##
    ## Find which y_comps are in toRemove
    ##
    hold = np.isin(y_comp, toRemove)              ## Identify the spots where we need to remove data (true class is in toRemove)
    
    
    ##
    ## Split x and y into known and unknown
    ##
    x_known = x[~hold,]    ## Subset x to knowns
    x_unknown = x[hold,]   ## Subset x to unknowns
    y_known = y[~hold,]    ## Subset y to knowns
    y_unknown = y[hold,]   ## Subset y to unknowns
    
    ##
    ## Remove unknown column from y
    ##
    y_known = np.delete(y_known, toRemove, axis = 1)       ## Remove the column indicies associated with the classes being removed
    y_unknown = np.delete(y_unknown, toRemove, axis = 1)   

    return(x_known, x_unknown, y_known, y_unknown)
    
    
##
## Function to evaluate results
##
##
## Params:
##
##   confidence - numpy array returned by evm.predict method
##   truth - numpy array of 1-hot ground truth labels
##   threshold - list of thresholds for which to evaluate the results
##
## Returns: 
## 
##    correct_rate - percent of predictions not called novel that are the correct class (note this means nothing when predicting truly unknown data)
##    novelty_rate - percent of predictions called novel
##
def eval(confidences, truth, thresholds):
    
    novelty_rate = []                            ## Store novelty rate
    correct_rate = []                            ## Store correct classification rate

    pred = np.argmax(confidences, axis = 1)      ## Find the predicions based on the confidences
    truth = np.argmax(truth, axis = 1)           ## Find the ground truth labels
    con = np.max(confidences, axis = 1)          ## Find the maximum confidence of the predictions (for each sample)

    for i in thresholds:
        novelty_rate.append(np.sum(con<i)/len(con))      ## con<i shows the times that the maximimum confidence is below the threshold

        correct = np.sum(pred[con>=i] == truth[con>=i])/np.sum(con>=i)
        correct_rate.append(correct)

    return(correct_rate, novelty_rate)
    