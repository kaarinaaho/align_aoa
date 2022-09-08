#! /home/kaarina/.conda/envs/align/bin/python
from pathlib import Path

import os,inspect
import pickle
import csv

import numpy as np
import pandas as pd
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)


def load_intersect_oldword_brettimg(
            img_id=0,
            wd_emb_path=os.path.join(parentdir,
                                    "assets/embeddings/glove.840B.300d.txt")
                                    ):

    img_embs = pickle.load(
        open(
            "/home/kaarina/projects/for_kaarina/results/z_{}_glove.p".format(
                img_id
            ),
            'rb'))
    img_embs = img_embs["z"]

    idx = pickle.load(
        open("/home/kaarina/projects/for_kaarina/mid_2_coidx.p", 'rb')
        )
    img_lab = pd.read_csv(
        "/home/kaarina/projects/train_embeddings/files/class-descriptions-boxable.csv",
        header=0)
    img_vocab = [
                img_lab.loc[img_lab["LabelName"] == x, "DisplayName"].iloc[0]
                for x in list(idx.keys())
                ]
    img_vocab = [x.lower() for x in img_vocab]  

    words = pd.read_table(wd_emb_path,
                          sep=" ", 
                          index_col=0,
                          header=None,
                          quoting=csv.QUOTE_NONE)
    wd_lab = list(words.index.values)
    wd_lab = [str(x).lower() for x in wd_lab]
    wd_embs = words.to_numpy()
    print("Loaded embs")

    # Load image vocab

    print("Loaded vocabs")

    joined = [x for x in list(img_vocab) if x in wd_lab]

    # Filter embeddings for items in both modalities
    wd_idx = [wd_lab.index(x) for x in joined]
    img_idx = [list(img_vocab).index(x) for x in joined]
    print("Filtered for intersection")

    # Sort by appropriate index
    wd_embs = wd_embs[wd_idx,:]
    img_embs = img_embs[img_idx,:]

    # Return intersections and vocab
    return wd_embs, img_embs, joined


def glove_word_and_glove_image(fp_intersect):
    """Load GloVe embeddings for words and images.
    The embeddings and vocabulary are returned in aligned order.
    Arguments:
        fp_intersect: A filepath to the intersect directory.
    Returns:
        z_0: The first embedding based on words.
        z_1: The second embedding based on images.
        vocab: A list of the intersection vocabulary.
    """
    fp_intersect_word_image = fp_intersect / Path(
        'intersect_glove.840b-openimage.box.p'
    )
    intersect_data = pickle.load(open(fp_intersect_word_image, 'rb'))
    z_0 = intersect_data['z_0']
    z_1 = intersect_data['z_1']
    vocab = list(intersect_data['vocab_intersect'])
    return np.squeeze(np.array(z_0)), np.squeeze(np.array(z_1)), vocab


def glove_word_and_glove_audio(fp_intersect):
    """Load GloVe embeddings for words and audio.
    The embeddings and vocabulary are returned in aligned order.
    Arguments:
        fp_intersect: A filepath to the intersect directory.
    Returns:
        z_0: The first embedding based on words.
        z_1: The second embedding based on audio.
        vocab: A list of the intersection vocabulary.
    """
    fp_intersect_word_image = fp_intersect / Path(
        'intersect_glove.840b-audioset.p'
    )
    intersect_data = pickle.load(open(fp_intersect_word_image, 'rb'))
    z_0 = intersect_data['z_0']
    z_1 = intersect_data['z_1']
    vocab = list(intersect_data['vocab_intersect'])
    return np.squeeze(np.array(z_0)), np.squeeze(np.array(z_1)), vocab


def glove_image_and_glove_audio(fp_intersect):

    """Load GloVe embeddings for images and audio.
    The embeddings and vocabulary are returned in aligned order.
    Arguments:
        fp_intersect: A filepath to the intersect directory.
    Returns:
        z_0: The first embedding based on words.
        z_1: The second embedding based on images.
        vocab_intersect: A list of the intersection vocabulary.
    """
    fp_intersect_word_image = fp_intersect / Path(
        'intersect_openimage.box-audioset.p'
    )
    intersect_data = pickle.load(open(fp_intersect_word_image, 'rb'))
    z_0 = intersect_data['z_0']
    z_1 = intersect_data['z_1']
    vocab = list(intersect_data['vocab_intersect'])
    return np.squeeze(np.array(z_0)), np.squeeze(np.array(z_1)), vocab
