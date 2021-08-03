import itertools as it
from multiprocessing import Pool, cpu_count

import libmr
import numpy as np


def set_cover_greedy(universe, subsets, cost=lambda x: 1.0):
    """Perform a greedy set cover

    This function performs a greedy set cover. 
        
    Args:
        universe: the instances that must be covered
        subsets: subsets of instances that can be used to cover the universe
        options (object): options object for the classifier
            
    Returns 
        cover_indices: indices of the subsets used to cover the universe
    """
    universe = set(universe)  # instances that must be covered
    subsets = [set(x) for x in subsets]  # get unique elements for each subset
    covered = set()  # a set for instances used to cover the universe
    cover_indices = []  # list to track index of subsets used to cover

    count = 0
  #  print('Fitting to a class')
    while covered != universe:
     #   print('Fitting to: ' + str(count))
     #   print(f'...{count}: {len(covered)} of {len(universe)}')

        # Find index of subset that would cover the most of the uncovered universe
        max_index = np.argmax(np.array([len(x - covered) for x in subsets]))

        # Add the best subset to the covered
        covered |= subsets[max_index]

        # Track the index of the best subset
        cover_indices.append(max_index)
        count += 1

    # print()
    return cover_indices


def set_cover(points, weibulls, cover_threshold, pdist_func, solver=set_cover_greedy):
    """Generic wrapper for set cover.

    Generic wrapper for set cover. Takes a solver function.
    Could do a Linear Programming approximation, but the
    default greedy method is bounded in polynomial time.
        
    Args:
        points: the instances that must be covered
        weibulls: distribution of instances
        cover_threshold: how close instances have to be to be "covered"
        pdist_func: how to measure distance between instances
        solver: how to get the set cover
            
    Returns 
        keep_indices: indices of the instances to keep in the EVM
    """
    universe = range(len(points))  # universe is all of the instances
    d_mat = pdist_func(points)  # find distance between points

    # Find probabilities that instances cover one another given distances and weibuls
    pool = Pool(cpu_count())
    probs = np.array(pool.map(weibull_eval_parallel, zip(d_mat, weibulls)))
    pool.close()
    pool.join()

    # make sure probabilities along diagonal are 1, ie probability of self is 100%
    np.fill_diagonal(probs, 1.0)

    # Find where probabilities are greater than threshold
    thresholded = zip(*np.where(probs >= cover_threshold))

    # Create subsets following probabilities and threshold
    subsets = {k: tuple(set(x[1] for x in v)) for k, v in it.groupby(thresholded, key=lambda x: x[0])}
    subsets = [subsets[i] for i in universe]

    # figure out what subsets are needed to cover the universe
    return solver(universe, subsets)


def reduce_model(points, weibulls, labels, cover_threshold, pdist_func, labels_to_reduce=None):
    """The EVM uses instance to define the boundaries of classes, this function removes unneeded instances to reduce
    the size of the model.
    :param points: the instances that must be covered
    :param weibulls: distribution of instances
    :param cover_threshold: how close instances have to be to be "covered"
    :param pdist_func: how to measure distance between instances
    :param labels_to_reduce: which classes should be reduced (default to all)
    :return: the points, weibuls, and labels of the reduced model
    """
    # no need to reduce
    if cover_threshold >= 1.0:
        return points, weibulls, labels

    # get classes of the dataset
    unique_labels = np.unique(labels)

    # if no labels to reduce, reduce all of them
    if labels_to_reduce is None:
        labels_to_reduce = unique_labels

    # get set of classes
    labels_to_reduce = set(labels_to_reduce)  # remove redundancies in labels to get set of classes
    keep = np.array([], dtype=int)  # store indices of instances being kept

    # reduce model for each class
    for label in unique_labels:
        indices = np.where(labels == label)  # find indices of class being reduced
        if label in labels_to_reduce:  # if it must be reduced, run set cover
            keep_ind = set_cover(points[indices], [weibulls[i] for i in indices[0]], cover_threshold, pdist_func)
            keep_idx = indices[0][keep_ind]
        else:
            keep_idx = indices[0]
        keep = np.concatenate((keep, keep_idx))

    # subset to reduced model
    points = points[keep]
    weibulls = [weibulls[i] for i in keep]
    labels = labels[keep]
    return points, weibulls, labels


def weibull_fit_parallel(args):
    """Fit weibull distributions for each class
   
    Args:
        args: really takes tailsize, dists, row, labels but only args for parallel

            
    Returns 
        points: points of the reduced model
        weibuls: weibuls of the reduced model
        labels: labels of the reduced model        
    """

    tailsize, dists, row, labels = args
    if len(dists[np.where(labels != labels[row])]) < tailsize:  # if tailsize > than the number of instances, will break
        tailsize = int(len(dists[np.where(labels != labels[row])]) * 0.9)  # override tailsize and shrink it if so

    # I don't really know what happens below.  But we somehow get weibulls out of it?
    nearest = np.partition(dists[np.where(labels != labels[row])], tailsize)
    mr = libmr.MR()

    mr.fit_low(nearest, tailsize)
    return str(mr)


def weibull_eval_parallel(args):
    """Evaluate weibulls
   
    Args:
        args: really takes dists and weibull params as strings

            
    Returns 
        probs: probability given distances and weibull distributions       
    """

    dists, weibull_params = args
    mr = libmr.load_from_string(weibull_params)
    probs = mr.w_score_vector(dists)
    return probs


def fuse_prob_for_label(prob_mat, num_to_fuse):
    """Average over probabilities to make predictions
    
    The EVM paper found it useful to average over k-largest
    probabilities when making a prediction (1-2% points)
   
    Args:
        prob_mat: probability matrix
        num_to_fuse: number of vectors to average over

            
    Returns 
        result: the averaged vectors    
    """

    result = np.average(np.partition(prob_mat, -num_to_fuse, axis=0)[-num_to_fuse:, :], axis=0)
    return result


def fit(X, y, tailsize, pdist_func, margin_scale):
    """Fit the EVM
   
    Args:
        X: training data
        y: training labels
        pdist_func: function to measure distance
        margin_scale: where to put margin between classes (probability of inclusion drops to 0 at margin...I think)
        
    Returns 
        weibulls: the weibulls that have been fit
    """

    ##
    ## Haven't taken the time to understand this
    ##
    d_mat = margin_scale * pdist_func(X)
    p = Pool(cpu_count())
    row_range = range(len(d_mat))
    args = zip(it.repeat(tailsize), d_mat, row_range, [y for i in row_range])
    weibulls = p.map(weibull_fit_parallel, args)
    p.close()
    p.join()
    return weibulls


def predict_evm(X, points, weibulls, labels, cdist_func, num_to_fuse):
    """Make predictions with the EVM
   
    Args:
        X: the instances to predict
        points: points making up the model
        weibulls: weibulls making up the model
        labels: labels making up the model
        cdist_func: function to measure distance (cdist takes a second array...what we are predicting)
        num_to_fuse: average prediction over this many vectors
        
    Returns 
        predicted_labels: predicted labels
        fused_probs: fused probabilities
    """

    d_mat = cdist_func(points, X).astype(np.float64)
    p = Pool(cpu_count())
    probs = np.array(p.map(weibull_eval_parallel, zip(d_mat, weibulls)))
    p.close()
    p.join()
    ulabels = np.unique(labels)
    fused_probs = []
    for ulabel in ulabels:
        fused_probs.append(fuse_prob_for_label(probs[np.where(labels == ulabel)], num_to_fuse))
    fused_probs = np.array(fused_probs)
    max_ind = np.argmax(fused_probs, axis=0)
    predicted_labels = ulabels[max_ind]
    confidence = fused_probs[max_ind]
    return predicted_labels, fused_probs
