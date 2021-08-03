from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
import numpy as np
import os
import keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.models import Model
import tensorflow


##
## Define a CNN
##
## Params:
##     
##    num_classes - number of classes in training data
##
## Returns:
##
##    model - model ready to train
##
#def define_model(num_classes):
 #   model = Sequential()
  #  model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))   ## Input assumes MNIST
    #model.add(MaxPooling2D((2, 2)))
   # model.add(Flatten())
    #model.add(Dense(100, activation='relu', kernel_initializer='he_uniform', name = 'features'))   ## This is our feature layer
   # model.add(Dense(num_classes, activation='softmax'))
    # compile model
  #  opt = SGD(lr=0.01, momentum=0.9)
   # model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
   # return(model)

##
## Create a feature extraction model
##
## Params:
## 
##    num_classes - number of classes in training data
##    x_train - numpy array of x's (images)
##    y_train - numpy array of 1-hot vector of y's (labels)
##    epochs - number of training epochs to run
##    batch_size - batch size
##    verbose - verbose
##
## Returns:
##
##     feature_model - model that can be used to extract features from x's (images)
##
#def feature_model():
  #      feature_model = tensorflow.keras.applications.ResNet50V2(weights='imagenet',
                                                                 #include_top=False)     
    
  #  return feature_model  

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
    