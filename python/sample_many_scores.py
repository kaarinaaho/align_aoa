#! /home/kaarina/.conda/envs/align/bin/python

import sys

sys.path.append(
    '/Users/apple/projects/Age_of_acquisition/python'
    )
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
rootdir =  os.path.dirname(parentdir)
sys.path.insert(0,parentdir) 

import utils
import numpy as np
import pandas as pd



def sample_alignment_scores(n_samples=5000, cand_set_size=None):
    # Load in all concepts
    vocab, wd_pw, img_pw, ___, ___ = utils.load_graphs_and_pw(
        q=0.1, dist_type="eu"
        )

    # Load AoA data, for the number of concepts acquired in each month
    # Load AoA data
    aoa_dat = utils.load_aoa_data()
    aoa_dat = aoa_dat.loc[aoa_dat["concept_i"].isin(vocab)]

    # Load concepts acquired by month
    ___, mean_concepts_acquired = utils.monthly_concepts(aoa_dat)


    # Sample sizes should bet the values in mean_concepts_acuired + 2, to match the probe pair test
    for month, ___ in mean_concepts_acquired.items():

        get_no = np.sum(
            [x
            for i, x in mean_concepts_acquired.items()
            if i <= month]) - 1

        for s in range(n_samples):
            if s % 100 == 0:
                print("month: {}, s: {}".format(month, s))

            # Emulate selection of candidate set:
            if cand_set_size is None:
                cands = [x for x in range(len(vocab))]
            else:
                cands = np.random.choice(
                    [x for x in range(len(vocab))],
                    cand_set_size, replace=False)

            # Emulate concept selection from candidate set
            ind_s = np.random.choice(cands, int(get_no)+2, replace=False)

            # Probs across remaining concepts
            rem = len(cands) - len(ind_s)
            ps = np.expand_dims(
                np.array([1/rem if i not in ind_s else 1 for i in cands]), 1
                )
            ps = np.matmul(ps, np.transpose(ps))

            score_right = utils.weighted_alignment_correlation_nontf(
                wd_pw[:, cands][cands, :], img_pw[:, cands][cands, :], ps
                )

            c_2 = list(cands)
            a, b = c_2.index(ind_s[-1]), c_2.index(ind_s[-2])
            c_2[b], c_2[a] = c_2[a], c_2[b]

            score_wrong = utils.weighted_alignment_correlation_nontf(
                wd_pw[:, cands][cands, :],
                img_pw[:, np.array(c_2)][np.array(c_2), :],
                ps)

            loss = score_wrong - score_right

            df = pd.DataFrame({
                "month":[month], "n_acq":[get_no+2], "loss":[loss]
            })
            df.to_csv(os.path.join(
                        rootdir, "assets/sample_pair_alignments/samples.csv"),
                    mode="a", header=None)

    samples = pd.read_csv(os.path.join(
                rootdir, "assets/sample_pair_alignments/samples.csv"),
                header=None, index_col=0)
    samples.columns = ["month", "n_acq", "loss"]
    samples = samples[["month", "loss"]].groupby("month").agg(["mean", "std"])
    samples.to_csv(os.path.join(rootdir, ))


def sample_MSE_losses(n_samples=5000, cand_set_size=None):
    # Load in all concepts
    vocab, ___, ___, ___, ___ = utils.load_graphs_and_pw(q=0.1)

    # Load AoA data, for the number of concepts acquired in each month
    # Load AoA data
    aoa_dat = utils.load_aoa_data()
    aoa_dat = aoa_dat.loc[aoa_dat["concept_i"].isin(vocab)]

    # Load concepts acquired by month
    ___, mean_concepts_acquired = utils.monthly_concepts(aoa_dat)

    # Sample sizes should bet the values in mean_concepts_acuired + 2, to match the probe pair test
    for month, ___ in mean_concepts_acquired.items():

        get_no = np.sum([
            x
            for i, x in mean_concepts_acquired.items()
            if i <= month]) - 1

        for s in range(n_samples):
            if s % 100 == 0:
                print(f"month: {month}, s: {s}")

            # Emulate selection of candidate set:
            if cand_set_size is None:
                cands = np.array([x for x in range(len(vocab))])
            else:
                cands = np.random.choice(
                    [x
                    for x in range(len(vocab))
                    ], cand_set_size, replace=False)

            # Emulate concept selection from candidate set
            ind_s = np.random.choice(cands, int(get_no), replace=False)

            # Probs across remaining concepts
            rem = len(cands) - len(ind_s)
            ps = np.array([1/rem if i not in ind_s else 1 for i in cands])


            y_true = [
                aoa_dat.loc[aoa_dat["concept_i"] == vocab[x], month].iloc[0]
                if vocab[x] in list(aoa_dat["concept_i"])
                else 0
                for x in cands]
            loss = np.mean(np.square(np.array(y_true) - np.array(ps)))


            df = pd.DataFrame({
                "month":[month], "n_acq":[get_no+1], "loss":[loss]
            })
            df.to_csv(os.path.join(rootdir,
                                    "assets/sample_mse_aoa/samples.csv"),
                                    mode="a", header=None)

    samples = pd.read_csv(os.path.join(
                                    rootdir,
                                    "assets/sample_mse_aoa/samples.csv"),
                            header=None, index_col=0)
    samples.columns = ["month", "n_acq", "loss"]
    samples = samples[["month", "loss"]].groupby("month").agg(["mean", "std"])
    samples.to_csv(os.path.join(
                            rootdir,
                            "assets/sample_mse_aoa/sample_summaries.csv"))


if __name__ == "__main__":

    sample_MSE_losses()
    sample_alignment_scores()
