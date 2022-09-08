 #! /home/kaarina/.conda/envs/align/bin/python

import sys

import os,sys,inspect
currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe()))
    )
parentdir = os.path.dirname(currentdir)
rootdir =  os.path.dirname(parentdir)
sys.path.insert(0,parentdir)

import utils
import numpy as np
import pandas as pd


def probe_pair_expt(n_probe_reps=100,
                    conditions=["synthProbIncAoA", "synthProbExcAoA"],
                    results_fn="results/probe_pair_results.csv",
                    months=None, test_idx=None):
    """Run forced choice experiment."""
    vocab, wd_pw, img_pw, __, __ = utils.load_graphs_and_pw(dist_type="eu")

    # Load AoA data
    aoa_dat = utils.load_aoa_data()
    aoa_dat = aoa_dat.loc[aoa_dat["concept_i"].isin(vocab)] # Filter for AoA items in vocab

    test_vocab = vocab
    if test_idx is not None:
        test_vocab = [vocab[x] for x in test_idx] # If only using some test pairs, reduce vocab set

    # If file doesn't exist, write headers in
    if not os.path.exists(results_fn):
        df = pd.DataFrame(columns=[
                    "month", "pid", "rep" ,"item_i", "item_j", "score_correct",
                    "score_incorrect", "correct_choice", "given", "probe",
                    "n_given"])
        df.to_csv(results_fn)
    else:
        df = pd.read_csv(results_fn, header=0, index_col=0)

    for knowledge in conditions:

        # Load file with knowledge condition sequences
        seqs = pd.read_csv(
            os.path.join(
                parentdir,
                f"results/sample_sequences/{knowledge}_sequences 2.csv"
                ),
            header=0, index_col=0
            )

        for pid in range(100):

            # Filter for pid's trajectory
            pid_seq = seqs.loc[seqs["pid"] == pid]

            if months is None:
                months = list(set(pid_seq["month"]))
            months.sort()

            for month in months:
                # Seed with concepts acquired before + in that month
                seed_concepts = pid_seq.loc[
                    pid_seq["month"] <= month, "concept"
                    ]

                # Get indices in array
                start_mapping = [vocab.index(x) for x in list(seed_concepts)]

                for probe_condition in ["AoA", "control"]:
                    print
                    (f"Pid: {pid}, month: {month}, knowledge: {knowledge}, "
                     + f"probe_condition: {probe_condition}"
                    )

                    # See if tests already in file
                    filt_curr = df.loc[
                        (df["probe"] == probe_condition)
                        & (df["given"] == knowledge)
                        & (df["n_given"] == len(seed_concepts))
                        & (df["pid"] == pid)]

                    complete_reps = len(set(filt_curr["rep"]))

                    # Pick up where file leaves off
                    if len(filt_curr) != n_probe_reps:
                        for rep in range(complete_reps, n_probe_reps):
                            if rep % 50 == 0:
                                print(f"pid: {pid}; rep {rep}")

                            if probe_condition == "AoA":
                                # Remaining vocab is untrained AoA vocab
                                remaining = [
                                    x for x in list(set(aoa_dat["concept_i"]))
                                    if x not in list(seed_concepts)
                                    and x in list(set(test_vocab))]

                            elif probe_condition == "control":
                                # Remaining vocab is untrained full vocab
                                remaining = [
                                    x for x in list(set(test_vocab))
                                    if x not in list(seed_concepts)]

                            test_points = np.random.choice(
                                remaining, 2, replace=False
                                )

                            # Learn a mapping for these two items
                            # (measure alignment correlation for two configs, take max)
                            test_idx = [
                                vocab.index(x) for x in list(test_points)
                                ]
                            correct_map = start_mapping + test_idx
                            incorrect_map = start_mapping + [
                                test_idx[1], test_idx[0]
                                ]

                            score_correct = utils.alignment_correlation(
                                wd_pw[correct_map, :][:, correct_map],
                                img_pw[correct_map, :][:, correct_map]
                                )
                            score_incorrect = utils.alignment_correlation(
                                wd_pw[correct_map, :][:, correct_map],
                                img_pw[incorrect_map, :][:, incorrect_map]
                                )

                            appendage = pd.DataFrame({
                                "month": [month],
                                "pid": [pid],
                                "rep": [rep],
                                "item_i": [test_points[0]],
                                "item_j": [test_points[1]],
                                "score_correct": [score_correct],
                                "score_incorrect": [score_incorrect],
                                "correct_choice": [score_correct > score_incorrect],
                                "given": [knowledge],
                                "probe": [probe_condition],
                                "n_given": [len(seed_concepts)],
                            })

                            appendage.to_csv(results_fn,
                                            mode="a", header=None)



if __name__ == "__main__":

    probe_pair_expt(
        results_fn=os.path.join(
            parentdir, "results/probe_pair/probe_pair_emp.csv"
            ),
        conditions=["AoA", "controlExcAoA", "controlIncAoA"],
                    months=None
                    )
