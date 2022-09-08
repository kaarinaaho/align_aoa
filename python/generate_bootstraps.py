#! /home/kaarina/.conda/envs/align/bin/python

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import numpy as np
import pandas as pd
import utils

"Generate bootstrap distributions from concept acquisition trajectories."

def generate_sequences(n_sequences, n_bootstraps, seq_fp):
    """Generate sequences of concepts sampled from AoA data."""
    # Quickload

    textfile = open(os.path.join(
                    parentdir, "assets/embeddings/vocab.txt"
                    ), "r")
    vocab = textfile.read().split('\n')

   # Load AoA data
    aoa_dat = utils.load_aoa_data()
    aoa_dat = aoa_dat.loc[aoa_dat["concept_i"].isin(vocab)]

    ___, mean_concepts_acquired = utils.monthly_concepts(aoa_dat)

    # Write headings to CSV
    df = pd.DataFrame(
        columns=["bootstrap", "pid", "month", "concept", "concept_id"])
    df.to_csv(seq_fp)

    # For sequence in n_sequence
    for b in range(n_bootstraps):

        # For each bootstrapped distribution, sample n sequences
        for pid in range(n_sequences):

            print(f"Distribution {b}: sequence {pid}/{n_sequences}")

            acquired_concepts = []

            for month, n_acq in mean_concepts_acquired.items():

                # Sample n_month_concepts from prob dist across remaining concepts
                remaining_concepts = [
                    vocab.index(x)
                    for x in aoa_dat["concept_i"]
                    if vocab.index(x) not in acquired_concepts
                    ]

                probs = np.array([
                    p for i, p in zip(aoa_dat["concept_i"], aoa_dat[month])
                    if vocab.index(i) in remaining_concepts
                    ])

                # Rescale probs to sum to 1
                probs = probs/sum(probs)
                learned_concepts = np.random.choice(
                    remaining_concepts, size=n_acq, p=probs, replace=False
                    )

                acquired_concepts = acquired_concepts + list(learned_concepts)

                # Append to df of sequence_id, month, concept, concept_id
                df = pd.DataFrame({
                    "bootstrap": [b]*n_acq,
                    "pid": [pid]*n_acq,
                    "month": [month]*n_acq,
                    "concept": [vocab[x] for x in learned_concepts],
                    "concept_id": learned_concepts
                })

                # Save condition as AoA or control
                df.to_csv(
                    seq_fp,
                    mode="a", header=None
                    )


def dists_from_seqs(seq_fp, dist_fp):
    """Obtain bootstrap distributions from sampled sequences."""

    # Read in CSV of sequences
    bootstraps = pd.read_csv(
                    seq_fp,
                    header=0, index_col=0
                    )
    n_sequences = len(bootstraps["pid"].unique())

    # Calculate concept probabilities by month across sequences
    bs = bootstraps.drop(columns=["concept_id"]).groupby(
        ["bootstrap", "month", "concept"]
        ).agg("count").reset_index()
    bs["pc"] = bs["pid"]/n_sequences
    bs = bs[["bootstrap", "month", "pc", "concept"]]

    bs = bs.pivot_table(
        index=["bootstrap", "month"],
        columns=["concept"],
        fill_value=0).reset_index()

    bs = bs.melt(
        id_vars=["bootstrap", "month"]
        )[
            ["bootstrap", "month", "concept", "value"]
            ].rename(columns={"value":"pc"}).reset_index()

    bs["month"] = [int(x) for x in bs["month"]]
    bs = bs.sort_values("month", ascending=True)

    bs["cumsum"] = bs.groupby(
                                ["concept", "bootstrap"]
                            )["pc"].transform(pd.Series.cumsum)
    bs = bs.drop(columns =["index"]).reset_index()
    bs.to_csv(dist_fp)


def gen_bootstrap_dists(n_sequences, n_bootstraps):
    """Generate and save bootstrap distributions."""

    seq_fn = "assets/bootstrap_dists/bootstrap_sequences.csv"
    seq_fp = os.path.join(parentdir, seq_fn)

    dist_fn = "assets/bootstrap_dists/bootstrap_distributions.csv"
    dist_fp = os.path.join(parentdir, dist_fn)

    generate_sequences(n_sequences, n_bootstraps, seq_fp)
    dists_from_seqs(seq_fp, dist_fp)


if __name__ == "__main__":

    gen_bootstrap_dists(50, 1000)
