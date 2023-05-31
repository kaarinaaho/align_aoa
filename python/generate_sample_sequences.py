#! /opt/anaconda3/envs/align/bin/python

import sys
from pathlib import Path

from numpy.core.fromnumeric import ptp
sys.path.append(
    '/Users/apple/projects/Age_of_acquisition/python'
    )
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import math
import pretrained_embeddings as emb
import utils
import mergeaoa as aoa
import numpy as np
import pandas as pd


class SeqGenerator:
    """Class to generate simulated concept learning trajectories."""
    def __init__(self, full_vocab, sequence_types=["AoA"]):
        """Init.

        - full_vocab: (list) list of words in the full vocab, to sample
        from
        - sequence_type: (list) sampling methods for sequences. List 
        | may contain any combination of "AoA", "controlIncAoA" and
        | "controlExcAoA"
        """
        if not all(x in ["AoA", "controlIncAoA", "controlExcAoA"]
                    for x in sequence_types):
            raise ValueError('Not all sequence_types are valid.')
  
        self.sequence_types = sequence_types

        # Load AoA data
        aoa_data = pd.read_csv(
            os.path.join(
                Path(__file__).parent.parent,
                "assets/aoa_data/aoa_data_EN.csv",
            ),
            header=0, index_col=0)
        aoa_data = aoa_data.rename(columns={"definition": "concept_i"})

        self.vocab = full_vocab

        # Filter for items in embedding vocab
        aoa_data = aoa_data.loc[aoa_data["concept_i"].isin(full_vocab)]
        self.aoa_data = aoa_data
        concepts_known, concepts_acquired = self._get_monthwise_concepts()
        self.concepts_known = concepts_known
        self.concepts_acquired = concepts_acquired


    def _get_monthwise_concepts(self):
        """Get avg num concepts known/acquired by month from AoA data.""" 
        concepts_known = {}
        concepts_acquired = {}
        prev_known = 0

        for col in [str(x) for x in range(16, 31)]:
            known = int(np.sum(self.aoa_data[col]))
            acquired = known - prev_known
            
            # Skip weird month where acquisition goes backwards
            if acquired > 0:
                concepts_known[col] = known
                concepts_acquired[col] = acquired
                prev_known = known
        
        return concepts_known, concepts_acquired


    def _get_item_probs(self, condition, acquired_concepts, month):
        """Load set of items and respective probabilities for sampling."""  
        # Sample n_month_concepts from prob dist across remaining concepts
        if condition == "AoA":
            # Get concept and probability from AoA data, if concept not known
            tups = [
                (self.vocab.index(x), p)
                for x, p
                in zip(self.aoa_data["concept_i"],self.aoa_data[month])
                if self.vocab.index(x) not in acquired_concepts
                ]
            remaining_concepts = [
                x[0] for x in tups
                ]
            probs = np.array([
                x[1] for x in tups
                ])
            
        else:
            if condition == "controlIncAoA":
                # Uniform probability across all unknown concepts
                remaining_concepts = [
                    i for i in range(len(self.vocab))
                    if i not in acquired_concepts
                    ]


            if condition == "controlExcAoA":
                # Uniform probability across unknown concepts not in AoA data
                remaining_concepts = [
                    i for i in range(len(self.vocab))
                    if (i not in acquired_concepts) 
                    and (self.vocab[i] not in list(self.aoa_data["concept_i"]))
                    ]

            probs = np.ones(len(remaining_concepts))

        # Normailse probs
        probs = probs/sum(probs)
        return remaining_concepts, probs


    def generate(self, save_folder, n_sequences=100):
        """Generate sequences."""

        for condition in self.sequence_types:

            save_fp = os.path.join(save_folder, f"{condition}_sample_sequenceCODETEST.csv")

            # Initialise results file
            pd.DataFrame(
                columns=["pid","condition","month","concept","concept_id"]
                ).to_csv(save_fp)

            for pid in range(n_sequences):
                print(f"{pid}/{n_sequences}")

                # Initialise sequence
                acquired_concepts = []

                for month, n_acq in self.concepts_acquired.items():

                    remaining_concepts, probs = self._get_item_probs(
                        condition, acquired_concepts, month
                    )
                    learned_concepts = list(np.random.choice(
                        remaining_concepts, size=n_acq, p=probs, replace=False
                        ))

                    acquired_concepts = acquired_concepts + learned_concepts

                    # Append to df of sequence_id, month, concept, concept_id
                    df = pd.DataFrame({
                        "pid": [pid]*n_acq,
                        "condition": [condition] * n_acq,
                        "month": [month]*n_acq,
                        "concept": [self.vocab[x] for x in learned_concepts],
                        "concept_id": learned_concepts
                    })
                    # Save condition as AoA or control
                    df.to_csv(
                        save_fp,
                        mode="a", 
                        header=None
                        )


if __name__ == "__main__":

    # Example
    ___, ___, vocab = emb.embs_quickload(n_word_dim=50)

    # Check for directories; make if not present
    if not os.path.exists(
        os.path.join(Path(__file__).parent.parent, "results"
        )):
        os.mkdir(os.path.join(Path(__file__).parent.parent, "results"))
    
    if not os.path.exists(
        os.path.join(Path(__file__).parent.parent, "results/sample_sequences"
        )):
        os.mkdir(
            os.path.join(
                Path(__file__).parent.parent,
                "results/sample_sequences"
                ))
    

    generator = SeqGenerator(vocab, sequence_types=["AoA", "controlIncAoA", "controlExcAoA"])
    
    generator.generate(save_folder=os.path.join(Path(__file__).parent.parent,"results/sample_sequences"),
                       n_sequences=100)
    
