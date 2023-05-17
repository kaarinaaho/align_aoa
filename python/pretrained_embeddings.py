#! /home/kaarina/.conda/envs/align/bin/python
from pathlib import Path

import os,inspect
import pickle
import csv
import utils
import numpy as np
import pandas as pd
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)


def embs_quickload(n_word_dim=300):
    # Quickload
    if n_word_dim==300:
        wdnimg = np.loadtxt(
            os.path.join(
                Path(__file__).parent.parent,
                "assets/embeddings/wdnimg.txt"
            ))
    elif n_word_dim==50:
        wdnimg = np.loadtxt(
            os.path.join(
                Path(__file__).parent.parent,
                "assets/embeddings/wdnimg_50.txt"
            ))
    imgnwd = np.loadtxt(
            os.path.join(
                Path(__file__).parent.parent,
                "assets/embeddings/imgnwd.txt"
            ))

    wdnimg = utils.scale_array(wdnimg, -1, 1)
    imgnwd = utils.scale_array(imgnwd, -1, 1)

    textfile = open(os.path.join(
                    Path(__file__).parent.parent,
                    "assets/embeddings/vocab.txt"), "r")
    vocab = textfile.read().split('\n')

    return wdnimg, imgnwd, vocab

