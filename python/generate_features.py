import sys
from pathlib import Path

sys.path.append(
    '/Users/apple/projects/Age_of_acquisition/python'
    )
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import utils
import numpy as np
import pandas as pd
import os 
import networkx as nx
from sklearn.metrics.pairwise import euclidean_distances
import pretrained_embeddings as emb


class FeatureGenerator:
    def __init__(self, emb1, emb2, vocab, sequences, q=0.1):

        self.seqs = sequences
        self.emb1 = emb1
        self.emb2 = emb2
        self.vocab = vocab
        self.pw_1 = utils.scale_pw(euclidean_distances(emb1))
        self.pw_2 = utils.scale_pw(euclidean_distances(emb2))
        
        graph1, graph2 = self._get_graphs(q)
        self.graph1 = graph1
        self.graph2 = graph2


    def _get_stripped_varnames(self, df):
        
        agnostic = list(set([x.replace("_1", "").replace("_2", "")
                    for x in df.columns
                    if x not in ["condition", "month", "pid", "thresh", "distance"]]))
        agnostic = list(set([x.replace("wd_", "").replace("img_", "")
                    for x in df.columns
                    if x not in ["condition", "month", "pid", "thresh", "distance"]]))
        agnostic = [x for x in agnostic if x not in ["x", "y"]]
        agnostic.sort()
            
        return agnostic


    def _get_graphs(self, q):

        graph1 = utils.retain_q_connections(self.pw_1, q)
        graph2 = utils.retain_q_connections(self.pw_2, q)

        return graph1, graph2


    def _get_modality_averages(self, df):
        # Get list of var names stripped of modality
        agnostic = self._get_stripped_varnames(df)

        # Get density at 
        densities = [x for x in agnostic if "density" in x]
        agnostic = [x for x in agnostic if x not in densities]
        agnostic = agnostic + densities

        for var in agnostic:
            cols = [x for x in df.columns if var in x]
            df[var] = df[cols].mean(axis=1)
            
        return df


    def _process_pairwise_matrices(self):
        
        # Sort pairwise matrices along one axis for easy computation
        pw1_sorted = self.pw_1.copy()
        pw1_sorted.sort(axis=1)
        pw2_sorted = self.pw_2.copy()
        pw2_sorted.sort(axis=1)

        var_names = ["min_dist_full", "max_dist_full", "mean_dist_full",
                    "min_dist_subs", "max_dist_subs", "mean_dist_subs", 
                    "dist_skew_full", "dist_skew_subs"]
        
        # Initialise results
        results = pd.DataFrame({})

        # Keep the two embeddings separate; repeat calculations for each
        labs = dict(zip([0,1], ["_1", "_2"]))
                
        for condition in self.seqs["condition"].unique():
            for seq in range(100):

                print("seq #: ", seq)

                for t in list(set(self.seqs["month"])):
                    
                    # Filter knowledge base for this month, for this sequence
                    df_filt = self.seqs.loc[
                            (self.seqs["pid"] == seq)
                            & (self.seqs["condition"] == condition)
                            & (self.seqs["month"] <= t)]
                    
                    # Get indices in pw matrices for items in knowledge base
                    idx = [self.vocab.index(x) for x in df_filt["concept"]]
                    
                    item_results = [condition, seq, t]
                    col_names = ["condition", "pid", "month"]
                    
                    # Measures based on pairwise distances
                    for i, mat in enumerate([pw1_sorted, pw2_sorted]):
                        
                        # Min, max, mean distance to all items in set for knowledge base
                        item_results += [np.mean(mat[idx, 1])]
                        item_results += [np.mean(mat[idx, -1])]
                        item_results += [np.mean(np.mean(mat[idx, :], axis=1))]

                        # Min, max, mean distance within knowledge base 
                        subs = mat[np.ix_(idx, idx)]

                        subs.sort(axis=1)
                        item_results += [np.mean(subs[:, 1])]
                        item_results += [np.mean(subs[:, -1])]
                        item_results += [np.mean(np.mean(subs[:, :], axis=1))]

                        # Skews (of full distances and subs distances)
                        item_results += [
                            (((item_results[5] - item_results[3])/(item_results[4] - item_results[3]))-0.5)/0.5
                            ]
                        item_results += [
                            (((item_results[8] - item_results[6])/(item_results[7] - item_results[6]))-0.5)/0.5
                            ]
                        
                        col_names = col_names + [x + str(labs[i]) for x in var_names]

                    # Measures of dimensional coverage
                    for i, embs in enumerate([self.emb1,  self.emb2]):

                        # Get coverage range in each dimension
                        full_rng = np.max(embs, axis=0) - np.min(embs, axis=0)

                        # Get sample coverage 
                        subs_rng = np.max(
                            embs[idx, :], axis=0) - np.min(embs[idx, :],
                            axis=0)

                        item_results += [
                            np.mean(subs_rng/full_rng)
                        ]

                        col_names = col_names +["coverage" + str(labs[i])]

                    appendage = pd.DataFrame([item_results], columns=col_names)
                    results = pd.concat([results, appendage])
                    
        for v in var_names:
            results[v] = results[[x for x in results.columns if v in x]].mean(axis=1)
        
        return results


    def _generate_basic_graph_measures(self):

        var_names = ["min_deg_full", "max_deg_full", "mean_deg_full",
                    "min_deg_subs", "max_deg_subs", "mean_deg_subs",
                    "deg_skew_full", "deg_skew_subs"]

        labs = dict(zip([0,1], ["_1", "_2"]))

        results = pd.DataFrame({})

        for condition in seqs["condition"].unique():
            for seq in range(100):
                print("seq #: ", seq)
                for month in list(set(seqs["month"])):
                    df_filt = seqs.loc[
                            (seqs["pid"] == seq)
                            & (seqs["condition"] == condition)
                            & (seqs["month"] <= month)]
                    idx = [self.vocab.index(x) for x in df_filt["concept"]]

                    item_results = [condition, seq, month]
                    col_names = ["condition", "pid", "month"]

                    for i, mat in enumerate([self.graph1, self.graph2]):

                        # Min, max, mean distance general
                        item_results += [np.min(np.sum(mat[idx, :], axis=1))/len(self.vocab)]
                        item_results += [np.max(np.sum(mat[idx, :], axis=1))/len(self.vocab)]
                        item_results += [np.mean(np.sum(mat[idx, :], axis=1))/len(self.vocab)]

                        # Min, max, mean distance within subset
                        # Min, max, mean distance general
                        item_results += [np.min(np.sum(mat[idx, :][:, idx], axis=1))/len(idx)]
                        item_results += [np.max(np.sum(mat[idx, :][:, idx], axis=1))/len(idx)]
                        item_results += [np.mean(np.sum(mat[idx, :][:, idx], axis=1))/len(idx)]

                        # Skews (of full distances and subs distances)
                        item_results += [
                            (((item_results[5] - item_results[3])/(item_results[4] - item_results[3]))-0.5)/0.5
                            ]
                        item_results += [
                            (((item_results[8] - item_results[6])/(item_results[7] - item_results[6]))-0.5)/0.5
                            ]

                        col_names = col_names + [x + str(labs[i]) for x in var_names]
                    
                    appendage = pd.DataFrame([item_results], columns=col_names)
                    results = pd.concat([results, appendage])

        for v in var_names:
            results[v] = results[[x for x in results.columns if v in x]].mean(axis=1)
        
        results.columns = [
            x
            if x not in ["condition", "pid", "month"]
            else x for x in results.columns]

        return results


    def generate_non_nx_measures(self, fp):

        dist_mets = self._process_pairwise_matrices()
        dist_mets = self._get_modality_averages(dist_mets)

        graph_mets = self._generate_basic_graph_measures()

        # Merge graph results with distance results
        dist_mets = dist_mets.merge(
            graph_mets,
            on=["condition", "pid", "month"])
        
        if not os.path.isfile(fp):
            pd.DataFrame(columns=dist_mets.columns).to_csv(fp)
        dist_mets.to_csv(fp, mode="a", header=None)


    def generate_nx_measures(self, save_fp):
        "For knowledge bases, obtain features which require NetworkX."
       
        # Load vocab, distance matrices and numpy graphs
        g1_nx = nx.DiGraph(
            self.graph1,
            )
        g2_nx = nx.DiGraph(
            self.graph2
            )

        conditions = list(self.seqs["condition"].unique())

        cols = ["condition", "pid", "n_acq", "month", "clustering_full",
                "clustering_subs", "betweenness_full", "betweenness_subs"]

        if not os.path.isfile(save_fp):
            header = pd.DataFrame(columns=cols)
            header.to_csv(save_fp)

        condition_l = []
        pids = []
        i_s = []
        months = []

        betweenness_subs_l = []
        clustering_subs_l = []
        betweenness_full_l = []
        clustering_full_l = []

        # Then for each sequence (at each timepoint), take the subgraph
        # Calculate a suite of relevant features for items in subgraph
        for c in conditions:
            for s in range(100):

                condition_l = []
                pids = []
                i_s = []
                months = []

                betweenness_subs1_l = []
                clustering_subs1_l = []
                betweenness_full1_l = []
                clustering_full1_l = []
                betweenness_subs2_l = []
                clustering_subs2_l = []
                betweenness_full2_l = []
                clustering_full2_l = []
                betweenness_subs_l = []
                clustering_subs_l = []
                betweenness_full_l = []
                clustering_full_l = []
                #for s in range(1):

                seq = self.seqs.loc[
                    (self.seqs["pid"] == s) & (self.seqs["condition"] == c)
                    ]
        
                for m in list(set(seq["month"])):
                    print(m)

                    condition_l.append(c)
                    i_s.append(0)
                    pids.append(s)
                    months.append(m)

                    # Get mean betweenness and clustering for all items in sample in full space
                    curr = [
                        self.vocab.index(x)
                        for x in list(seq.loc[seq["month"] <= m, "concept"])
                        ]

                    clustering_full1 = np.mean([
                        nx.clustering(g1_nx, nodes=curr)[x] for x in curr
                        ])
                    clustering_full2 = np.mean([
                        nx.clustering(g2_nx, nodes=curr)[x] for x in curr
                        ])
                    clustering_full = (clustering_full1 + clustering_full2)/2

                    nx_bw_1 = nx.betweenness_centrality(g1_nx)
                    nx_bw_2 = nx.betweenness_centrality(g2_nx)
                    betweenness_full1 = np.mean([nx_bw_1[x] for x in curr])
                    betweenness_full2 = np.mean([nx_bw_2[x] for x in curr])
                    betweenness_full = (betweenness_full1 + betweenness_full2)/2
                    
                    G1_subs = g1_nx.subgraph(curr)
                    G2_subs = g2_nx.subgraph(curr)

                    clustering_subs1 = np.mean([nx.clustering(G1_subs, nodes=curr)[x] for x in curr])
                    clustering_subs2 =np.mean([nx.clustering(G2_subs, nodes=curr)[x] for x in curr])
                    clustering_subs = (clustering_subs1 + clustering_subs2)/2

                    nx_bw_1 = nx.betweenness_centrality(G1_subs)
                    nx_bw_2 = nx.betweenness_centrality(G2_subs)
                    betweenness_subs1 = np.mean([nx_bw_1[x] for x in curr])
                    betweenness_subs2 = np.mean([nx_bw_2[x] for x in curr])
                    betweenness_subs = (betweenness_subs1 + betweenness_subs2)/2
                        
                    # get mean betweenness and clustering for items in sample within sample
                    clustering_full1_l.append(clustering_full1)
                    clustering_subs1_l.append(clustering_subs1)
                    betweenness_full1_l.append(betweenness_full1)
                    betweenness_subs1_l.append(betweenness_subs1)
                    clustering_full2_l.append(clustering_full2)
                    clustering_subs2_l.append(clustering_subs2)
                    betweenness_full2_l.append(betweenness_full2)
                    betweenness_subs2_l.append(betweenness_subs2)
                    clustering_full_l.append(clustering_full)
                    clustering_subs_l.append(clustering_subs)
                    betweenness_full_l.append(betweenness_full)
                    betweenness_subs_l.append(betweenness_subs)

                df = pd.DataFrame({
                    "condition": condition_l,
                    "pid":pids,
                    "n_acq": i_s,
                    "month": months,
                    "clustering_full_wd": clustering_full1_l,
                    "clustering_full_img": clustering_full2_l,
                    "clustering_full": clustering_full_l,
                    "clustering_subs_wd": clustering_subs1_l,
                    "clustering_subs_img": clustering_subs2_l,
                    "clustering_subs": clustering_subs_l,
                    "betweenness_full_wd": betweenness_full1_l,
                    "betweenness_full_img": betweenness_full2_l,
                    "betweenness_full": betweenness_full_l,
                    "betweenness_subs_wd": betweenness_subs1_l,
                    "betweenness_subs_img": betweenness_subs2_l,
                    "betweenness_subs": betweenness_subs_l,
                })

                df.to_csv(save_fp, mode="a", header=None)


if __name__ == "__main__":

    control_seqs_exc = pd.read_csv(
        os.path.join(
            Path(__file__).parent.parent,
            "results/sample_sequences/controlExcAoA_sequences.csv"
            ),
            header=0, index_col=0
    )

    aoa_seqs = pd.read_csv(
        os.path.join(
            Path(__file__).parent.parent,
            "results/sample_sequences/AoA_sequences.csv"
            ),
            header=0, index_col=0
    )

    if not os.path.exists(
        os.path.join(Path(__file__).parent.parent, "results/seq_features"
        )):
        os.mkdir(
            os.path.join(
                Path(__file__).parent.parent,
                "results/seq_features"
                ))

    seqs = pd.concat([
        aoa_seqs,
        control_seqs_exc
        ])

    emb1, emb2, vocab = emb.embs_quickload(n_word_dim=50)

    feature_gen = FeatureGenerator(
        emb1, emb2, vocab, seqs
        )
    
    feature_gen.generate_non_nx_measures(
        os.path.join(
                Path(__file__).parent.parent,
                "results/seq_features/sequence_non_nx_mets_monthwise.csv"
                )
    )
    """
    feature_gen.generate_nx_measures(
                os.path.join(
                Path(__file__).parent.parent,
                "results/seq_features/sequence_nx_mets_monthwise.csv"
                )
    )
    """
