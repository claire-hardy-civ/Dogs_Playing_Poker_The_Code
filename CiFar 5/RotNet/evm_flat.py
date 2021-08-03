import numpy as np
import sklearn.metrics.pairwise

import evm_enabler
import general_model as gm


class EVMOptions():
    """Options class for an EVM classifier

    The options class allows us to set custom options for each classifier
    and still pass these instructions to a General Model.  For more information
    about the EVM visit the following: https://arxiv.org/pdf/1506.06112.pdf,
    https://github.com/EMRResearch/ExtremeValueMachine
    
    Attributes:
        threshold (double): threshold used to call something novel
        tailsize (int): the number of extreme vectors allowed for the weibull distributions
        cover_threshold (double): probabilistic threshold to designate redundancy between points. Paper sets at 0.5
        for all experiments
        dist_func (string): name of function used to calculate distance (value: 'cosine' or 'euclidean')
        num_to_fuse (int): when picking y* need to find max probability.  Num_to_fuse looks at average max prob for
        top 'num_to_fuse'. Paper found increase of 1-2 percentage points by tuning this.
        margin_scale (double): Default 0.5 where to put margin between points
        
    """

    def __init__(self, tailsize, cover_threshold, dist_func, num_to_fuse, margin_scale=0.5):
        self.tailsize = tailsize
        self.cover_threshold = cover_threshold
        self.dist_func = dist_func
        self.num_to_fuse = num_to_fuse
        self.margin_scale = margin_scale


class ExtremeValueMachine(gm.General_Model):
    """Class for the flat EVM classifier

    This class implements the General Model Abstract class.
    The result is an EVM classifier that 
    has no hierarchy.  It has the options as defined in
    the options object.  The init builds and compiles 
    the model.
    
    Attributes:
        name (string): the name of the classifier
        options (object): options object for the classifier
    """

    def __init__(self, name, options):

        self.name = name
        self.trained = False

        ##
        ## EVM parameters
        ##
        self.tailsize = options.tailsize  ## <- number of extreme vectors allowed
        self.cover_threshold = options.cover_threshold
        self.dist_func = options.dist_func
        self.num_to_fuse = options.num_to_fuse
        self.margin_scale = options.margin_scale

        ##
        ## Items for the model
        ##
        self.points = None
        self.weibulls = None
        self.labels = None

        ##
        ## Have to handle distances somewhere
        ##
        def euclidean_cdist(X, Y):
            return sklearn.metrics.pairwise.pairwise_distances(X, Y, metric="euclidean", n_jobs=1)

        def euclidean_pdist(X):
            return sklearn.metrics.pairwise.pairwise_distances(X, metric="euclidean", n_jobs=1)

        def cosine_cdist(X, Y):
            return sklearn.metrics.pairwise.pairwise_distances(X, Y, metric="cosine", n_jobs=1)

        def cosine_pdist(X):
            return sklearn.metrics.pairwise.pairwise_distances(X, metric="cosine", n_jobs=1)

        dist_func_lookup = {
            "cosine": {"cdist": cosine_cdist,
                       "pdist": cosine_pdist},

            "euclidean": {"cdist": euclidean_cdist,
                          "pdist": euclidean_pdist}
        }

        self.cdist_func = dist_func_lookup[self.dist_func]["cdist"]
        self.pdist_func = dist_func_lookup[self.dist_func]["pdist"]

    def train(self, x, y):

        """Train the the EVM classifier

        This function trains the EVM classifier.
        
        Args:
            train (list): [x_train, y_train] numpy arrays (y_train NOT one-hot)
            val (list): [x_val, y_val] numpy arrays (y_val NOT one-hot)
            options (object): options object for the classifier
            
        """
        ##
        ## Separate train and test
        ##

        x_train = x.astype('double')
        y_train = np.argmax(y, axis = 1)

        ##
        ## Fit the model (the model ends up being points, weibulls and labels)
        ## See https://arxiv.org/pdf/1506.06112.pdf for details
        ##
        self.weibulls = evm_enabler.fit(x_train, y_train, self.tailsize, self.pdist_func, self.margin_scale)
        self.points, self.weibulls, self.labels = evm_enabler.reduce_model(x_train, self.weibulls, y_train,
                                                                           self.cover_threshold, self.pdist_func)
        self.trained = True

    def confidence(self, pool):

        """Report confidence of predictions of the pool
        
        Args:
            pool (numpy.ndarray): instances to predict
            
        Returns 
            results: array of prediction confidences for the pool
        """
        if (self.trained):
            predictions, probs = evm_enabler.predict_evm(pool, self.points, self.weibulls, self.labels, self.cdist_func,
                                                         self.num_to_fuse)  ## Predict the pool

            return np.transpose(probs)
        else:
            print('Not trained! (prediction)')
