import os, sys, inspect
parentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

sys.path.insert(0,parentdir) 

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
#import tensorflow as tf
import scipy as sp



def scale_pw(pw, eps=1e-10):
    """Scale pw distance measure to be between eps and 1"""
    pw = pw - (np.min(pw) - eps) # E.g -(cosine sim) will not have min 0
    pw = pw/np.max(pw)
    return pw


def retain_q_connections(pw_dists, q):
    """Generate graph matrix where edges are q*100% of closest relationships."""
    # Get upper triangular
    pw_dists = np.triu(pw_dists)

    # Set self distances/tril to nan
    pw_dists[pw_dists == 0] = np.nan
    
    # Get quantile q of flattened array
    thresh = np.nanquantile(pw_dists, q)
    
    # First: construct a graph with a given threshold
    graph_in = np.where((pw_dists < thresh) & (pw_dists != np.nan), 1, 0)
    
    return graph_in


def load_graphs_and_pw(q=0.1, dist_type="eu"):

    # Quickload
    wdnimg = np.loadtxt(os.path.join(
                            os.path.dirname(currentdir),
                            "assets/embeddings/wdnimg.txt"))
    imgnwd = np.loadtxt(os.path.join(
                            os.path.dirname(currentdir),
                            "assets/embeddings/imgnwd.txt"))

    textfile = open(os.path.join(
                            os.path.dirname(currentdir),
                            "assets/embeddings/vocab.txt"), "r")
    vocab = textfile.read().split('\n')

    if dist_type == "cos":
        img_pw = -cosine_similarity(imgnwd, imgnwd)
        wd_pw = -cosine_similarity(wdnimg, wdnimg)
    
    else:
        img_pw = pairwise_distance(imgnwd, imgnwd)
        wd_pw = pairwise_distance(wdnimg, wdnimg) 


    # Generate a graph for word and image spaces 
    G_img = retain_q_connections(img_pw, q)
    img_pw = scale_pw(img_pw)

    G_wd = retain_q_connections(wd_pw, q)
    wd_pw = scale_pw(wd_pw)

    return vocab, wd_pw, img_pw, G_wd, G_img


def load_aoa_data(
        aoa_fp="assets/aoa_data/aoa_data_EN.csv",
        acquisition_threshold=None):
    """

    """

    # Import aoa data and match column names
    aoa = pd.read_csv(os.path.join(os.path.dirname(parentdir), aoa_fp), header=0, index_col=0)
    aoa = aoa.rename(columns={"definition": "concept_i"})

    if acquisition_threshold is not None:
        # Find age (in months) at which threshold acquisition is achieved
        mins = pd.melt(
            aoa, id_vars=["concept_i", "category", "type"],
            var_name="age", value_name="percent"
            )
        mins["above_thresh"] = mins["percent"] >= acquisition_threshold
        mins = mins[["concept_i", "age"]][mins["above_thresh"] == True]
        mins = mins.groupby("concept_i").agg("min")
        mins = mins.rename(columns={"age": "age_thresh_ac"})

        aoa = aoa.merge(mins, how="inner", on="concept_i")
        aoa["thresh"] = acquisition_threshold
        aoa = aoa.sort_values(by=["age_thresh_ac"])
        aoa.reset_index(inplace=True, drop=True)

    return aoa


def monthly_concepts(merged):
    """Load the mean number of object concepts acquired per month."""

    # Get avg number of concepts known in each month 
    mean_concepts_known = {}
    mean_concepts_acquired = {}
    prev_known = 0
    for col in [str(x) for x in range(16, 31)]:
        known = int(np.sum(merged[col]))
        acquired = known - prev_known
        
        # Skip weird month where acquisition goes backwards
        if acquired > 0:
            mean_concepts_known[col] = known
            mean_concepts_acquired[col] = acquired
            prev_known = known

    return mean_concepts_known, mean_concepts_acquired



def weighted_alignment_correlation_tf(pairwise_A, pairwise_B, w, f=None):
    """
    Calculate alignment correlation between two systems.

    Assumes systems are in the same space and inputted aligned
    """
    # Index of upper triangular matrices
    #idx_upper = tf.convert_to_tensor(np.triu_indices(pairwise_A.numpy().shape[0], 1))

    # Pairwise distance matrix between system A and system B
    #r_s = weighted_corr(tf.gather(pairwise_A, indices=idx_upper), tf.gather(pairwise_B, indices=idx_upper), w)
    r_s = weighted_corr(pairwise_A, pairwise_B, w)


    return r_s


# def weighted_corr(x, y, w):

#     # get upper triangular
#     #idx = np.triu_indices(np.shape(x)[0])
#     #idx0 = tf.constant(idx[0],  dtype="int32")
#     #idx1 = tf.constant(idx[1],  dtype="int32")

#     #x = tf.gather(tf.gather(x, idx0), idx1, axis=1, batch_dims=1)
#     #y = tf.gather(tf.gather(y, idx0), idx1, axis=1, batch_dims=1)
#     #w = tf.gather(tf.gather(w, idx0), idx1, axis=1, batch_dims=1)

#     c_xy = get_weighted_cov(x, y, w)
#     c_xx = get_weighted_cov(x, x, w)
#     c_yy = get_weighted_cov(y, y, w)

#     return tf.divide(c_xy, tf.math.sqrt(tf.multiply(c_xx, c_yy)))


# def get_weighted_cov(x, y, w):

#     m_x = get_weighted_mean(x, w)
#     m_y = get_weighted_mean(y, w)
#     num = tf.reduce_sum(tf.multiply(tf.multiply(w, (x-m_x)), (y-m_y)))
#     return tf.divide(num, tf.reduce_sum(w))


# def get_weighted_mean(x, w):
#     return tf.divide(tf.reduce_sum(tf.multiply(x, w)), tf.reduce_sum(w))


def alignment_correlation(pairwise_A, pairwise_B, f=None):
    """
    Calculate alignment correlation between two systems.

    Assumes systems are in the same space and inputted aligned
    """
    # Index of upper triangular matrices
    idx_upper = np.triu_indices(pairwise_A.shape[0], 1)

    # Take upper diagonal of corresponding sim matrices for A->B
    vec_A = pairwise_A[idx_upper]
    vec_B = pairwise_B[idx_upper]

    # Spearman correlation
    r_s = sp.stats.spearmanr(vec_A, vec_B)[0]

    return r_s


def pairwise_distance(systemA, systemB, norm="l2"):
    """
    Calculate pairwise distances between points in two systems.

    Args:
    - systemA and systemB: nxd
    """
    n = systemA.shape[0]

    if norm=="l2":    
        B_transpose = np.transpose(systemB)
            
        inner = -2 * np.matmul(systemA, B_transpose)
        
        A_squares = np.sum(
            np.square(systemA), axis=-1
            )
        A_squares = np.transpose(np.tile(A_squares, (n,1)))
            
        B_squares = np.transpose(
            np.sum(np.square(systemB), axis=-1)
            )   
        B_squares = np.tile(B_squares, (n, 1))

        pairwise_distances = np.sqrt(
            np.abs(
                inner + A_squares + B_squares
                )
        )

    elif norm=="l1":
        A_tile = np.stack([systemA]*systemB.shape[0], axis=1)
        B_tile = np.stack([systemB]*systemA.shape[0], axis=0)

        diffs = np.abs(np.subtract(A_tile, B_tile))
        pairwise_distances = np.sum(diffs, axis=-1)
    
    return pairwise_distances


# def pairwise_dist_tf(A, B):  
#     """
#     Computes pairwise distances between each elements of A and each elements of B.
#     Args:
#     A,    [m,d] matrix
#     B,    [n,d] matrix
#     Returns:
#     D,    [m,n] matrix of pairwise distances
#     """
#     # squared norms of each row in A and B
#     na = tf.reduce_sum(tf.square(A), 1)
#     nb = tf.reduce_sum(tf.square(B), 1)

#     # na as a row and nb as a co"lumn vectors
#     na = tf.reshape(na, [-1, 1])
#     nb = tf.reshape(nb, [1, -1])

#     # return pairwise euclidead difference matrix
#     D = tf.sqrt(tf.maximum(na - 2*tf.matmul(A, B, False, True) + nb, 0.0))
#     return D


def weighted_mean(x, w):
    "Calculate mean of x weighted by w."
    return np.sum(x * w)/np.sum(w) 


def weighted_cov(x, y, w):
    "Get weighted covariance of x and y, weighted by w."
    m_x = weighted_mean(x, w)
    m_y = weighted_mean(y, w)
    num = np.sum(w * (x-m_x) *  (y-m_y))
    return num/np.sum(w)  


def weighted_corr_nontf(x, y, w):
    "Get weighted correlation between x and y, weighted by w"
    c_xy = weighted_cov(x, y, w)
    c_xx = weighted_cov(x, x, w)
    c_yy = weighted_cov(y, y, w)

    return c_xy/np.sqrt(c_xx * c_yy)


def weighted_alignment_correlation_nontf(pairwiseA, pairwiseB, w, f=None):
    """
    Calculate alignment correlation between two systems.

    Assumes systems are in the same space and inputted aligned
    """
    # Index of upper triangular matrices
    idx_upper = np.triu_indices(pairwiseA.shape[0], 1)
    pairwiseA = pairwiseA[idx_upper]
    pairwiseB = pairwiseB[idx_upper]
    w = w[idx_upper]

    # Pairwise distance matrix between system A and system B
    r_s = weighted_corr_nontf(pairwiseA, pairwiseB, w)

    return r_s

"""
@tf.function
def get_rank(y):
    rank = tf.cast(tf.argsort(
        tf.argsort(y, axis=-1, direction="ASCENDING"), axis=-1),
        dtype=tf.float32) #+1 #+1 to get the rank starting in 1 instead of 0
    return rank
"""

if __name__ == "__main__":

    """
    mat = tf.constant(np.array([[0,1,2], [3,4,5], [6, 7, 8]]))

    idx = np.triu_indices(3)

    print(tf.gather(tf.gather(mat, idx[0]), idx[1], axis=1, batch_dims=1))
    """
