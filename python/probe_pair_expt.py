 #! /home/kaarina/.conda/envs/align/bin/python

import sys
from pathlib import Path
import os,sys,inspect
currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe()))
    )
parentdir = os.path.dirname(currentdir)
rootdir =  os.path.dirname(parentdir)
sys.path.insert(0,parentdir)

from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

import utils
import numpy as np
import pandas as pd
import pretrained_embeddings as emb


class ProbePairExpt:
    def __init__(self, emb1, emb2, vocab1, vocab2):
        self.emb1_raw = emb1
        self.emb2_raw = emb2
        
        # Get vocab of items in intersection between embeddings
        self.intersect_vocab = [x for x in vocab1 if x in vocab2]
        self.idx1 = [vocab1.index(x) for x in self.intersect_vocab]
        self.idx2 = [vocab2.index(x) for x in self.intersect_vocab]

        # Reindex embeddings s.t they align
        self.emb1_idxed = emb1[self.idx1]
        self.emb2_idxed = emb2[self.idx2]

        # Pairwise
        self.pw1 = euclidean_distances(self.emb1_idxed)
        self.pw2 = euclidean_distances(self.emb2_idxed)

        # Load AoA data
        aoa_dat = utils.load_aoa_data()
        # Filter for AoA items in vocab
        aoa_dat = aoa_dat.loc[aoa_dat["concept_i"].isin(self.intersect_vocab)]
        self.aoa_dat = aoa_dat


    def run(self, knowledge_template_fp, results_fp,
            knowledge_conditions=["AoA"], n_probe_reps=100, months=None):

        """Run forced choice experiment."""

        # If file doesn't exist, write headers to file
        if not os.path.exists(results_fp):
            df = pd.DataFrame(columns=[
                        "month", "pid", "rep" ,"item_i", "item_j",
                        "score_correct", "score_incorrect", "correct_choice",
                        "given", "probe", "n_given"])
            df.to_csv(results_fp)

        for knowledge in knowledge_conditions:

            # Load file with knowledge condition sequences
            seqs = pd.read_csv(
                knowledge_template_fp.format(knowledge),
                header=0, index_col=0
                )

            for pid in list(set(seqs["pid"])):

                # Filter for pid's trajectory
                pid_seq = seqs.loc[seqs["pid"] == pid]

                if months is None:
                    # If months unspecified, use all
                    months = list(set(pid_seq["month"]))
                else:
                    if not all(x in list(set(pid_seq["month"]))
                               for x in months):
                        raise ValueError("Given months must be in every sequence.")
                months.sort()

                for month in months:
                    # Seed with concepts acquired before + in that month
                    seed_concepts = pid_seq.loc[
                        pid_seq["month"] <= month, "concept"
                        ]

                    # Get indices in array
                    start_mapping = [
                        self.intersect_vocab.index(x)
                        for x in list(seed_concepts)
                        ]

                    for probe_condition in ["AoA", "control"]:
                        print(
                        f"Pid: {pid}, month: {month}, knowledge: {knowledge}, "
                        + f"probe_condition: {probe_condition}"
                        )

                        for rep in range(n_probe_reps):
                            if rep % 50 == 0:
                                print(f"pid: {pid}; rep {rep}")

                            if probe_condition == "AoA":
                                # Remaining vocab is untrained AoA vocab
                                remaining = [
                                    x for x
                                    in list(set(self.aoa_dat["concept_i"]))
                                    if x not in list(seed_concepts)]

                            elif probe_condition == "control":
                                # Remaining vocab is untrained full vocab
                                remaining = [
                                    x for x in self.intersect_vocab
                                    if x not in list(seed_concepts)]

                            test_points = np.random.choice(
                                remaining, 2, replace=False
                                )

                            # Learn a mapping for these two items
                            # (measure alignment correlation for two configs, take max)
                            test_idx = [
                                self.intersect_vocab.index(x)
                                for x in list(test_points)
                                ]
                            correct_map = start_mapping + test_idx
                            incorrect_map = start_mapping + [
                                test_idx[1], test_idx[0]
                                ]

                            # Evaluate alignment score for correct vs. incorrect
                            score_correct = utils.alignment_correlation(
                                self.pw1[correct_map, :][:, correct_map],
                                self.pw2[correct_map, :][:, correct_map]
                                )
                            score_incorrect = utils.alignment_correlation(
                                self.pw1[correct_map, :][:, correct_map],
                                self.pw2[incorrect_map, :][:, incorrect_map]
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

                            appendage.to_csv(results_fp,
                                            mode="a", header=None)


if __name__ == "__main__":

    # Run probe pair experiment with child-directed embeddings - are the results replicated?
    wdnimg, imgnwd, vocab1 = emb.embs_quickload(n_word_dim=50)

    expt = ProbePairExpt(imgnwd, wdnimg, vocab1, vocab1)
    expt.run(
        os.path.join(
            Path(__file__).parent.parent,
            "/Users/apple/Documents/GitHub/align_aoa/results/sample_sequences/{}_sequences.csv"),
            "/Users/apple/Documents/GitHub/align_aoa/results/probe_pair/probe_pair_emp.csv",
            ["AoA", "controlIncAoA", "controlExcAoA"]
    )
