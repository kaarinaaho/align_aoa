import sys

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


def get_stripped_varnames(df, strip=["modality"]):
    
    if "modality" in strip:
        agnostic = list(set([x.replace("wd_", "").replace("img_", "")
                    for x in df.columns
                    if x not in ["condition", "month", "pid", "thresh", "distance"]]))
        agnostic = [x for x in agnostic if x not in ["x", "y"]]
        agnostic.sort()
        
    if "quantile" in strip:
        agnostic = list(set([
            x.rsplit("_", 1)[0]
            for x in df.columns 
            if x not in ["condition", "month", "pid", "thresh", "distance"]]))
        agnostic.sort()
        
    return agnostic


def get_modality_averages(df):
    # Get list of var names stripped of modality
    agnostic = get_stripped_varnames(df, strip=["modality"])

    # Get density at end
    densities = [x for x in agnostic if "density" in x]
    agnostic = [x for x in agnostic if x not in densities]
    agnostic = agnostic + densities

    for var in agnostic:
        cols = [x for x in df.columns if var in x]
        df[var] = df[cols].mean(axis=1)
        
    return df


def process_pairwise_matrices(seqs, vocab, prefix="eu"):
    
    vocab, wd_pw, img_pw, __, __ = utils.load_graphs_and_pw(
                                            q=0.1, dist_type="eu")

    wd_pw_c = wd_pw.copy()
    img_pw_c = img_pw.copy()
    wd_pw_c.sort(axis=1)
    img_pw_c.sort(axis=1)
    
    var_names = ["min_dist_full", "max_dist_full", "mean_dist_full",
                 "min_dist_subs", "max_dist_subs", "mean_dist_subs", 
                 "dist_skew_full", "dist_skew_subs"]
    
    labs = dict(zip([0,1], ["wd_", "img_"]))
    
    results = pd.DataFrame({})
    
    for condition in seqs["condition"].unique():
        for seq in range(100):
            print("seq #: ", seq)
            for month in list(set(seqs["month"])):
                df_filt = seqs.loc[
                        (seqs["pid"] == seq)
                        & (seqs["condition"] == condition)
                        & (seqs["month"] <= month)]
                idx = [vocab.index(x) for x in df_filt["concept"]]
                
                item_results = [condition, seq, month]
                col_names = ["condition", "pid", "month"]
                
                for i, mat in enumerate([wd_pw_c, img_pw_c]):
                    
                    # Min, max, mean distance general
                    item_results += [np.mean(mat[idx, 1])]
                    item_results += [np.mean(mat[idx, -1])]
                    item_results += [np.mean(np.mean(mat[idx, :], axis=1))]

                    # Min, max, mean distance within subset
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
                    
                    
                    col_names = col_names + [str(labs[i]) + x for x in var_names]
                appendage = pd.DataFrame([item_results], columns=col_names)
                results = pd.concat([results, appendage])
                
    for v in var_names:
        results[v] = results[[x for x in results.columns if v in x]].mean(axis=1)
    results.columns = [
        prefix + x
        if x not in ["condition", "pid", "month"]
        else x for x in results.columns]
    print(results)
    return results


def generate_non_nx_measures(seqs):

    # Quickload
    wdnimg = np.loadtxt(os.path.join(
                            os.path.dirname(currentdir),
                            "assets/embeddings/wdnimg.txt"))
    imgnwd = np.loadtxt(os.path.join(
                            os.path.dirname(currentdir),
                            "assets/embeddings/imgnwd.txt"))

    vocab, wd_pw, img_pw, G_wd, G_img = utils.load_graphs_and_pw(
                                                q=0.1, dist_type="eu")
    
    #seqs["acq_id"] = [i for i in range(118)] * 400


    dist_mets = process_pairwise_matrices(seqs, vocab, prefix="")

    dist_mets = get_modality_averages(dist_mets)

    # Get degrees
        
    wd_pw_c = wd_pw.copy()
    img_pw_c = img_pw.copy()

    var_names = ["min_deg_full", "max_deg_full", "mean_deg_full",
                "min_deg_subs", "max_deg_subs", "mean_deg_subs", 
                "deg_skew_full", "deg_skew_subs"]

    labs = dict(zip([0,1], ["wd_", "img_"]))

    results = pd.DataFrame({})

    for condition in seqs["condition"].unique():
        for seq in range(100):
            print("seq #: ", seq)
            for month in list(set(seqs["month"])):
                df_filt = seqs.loc[
                        (seqs["pid"] == seq)
                        & (seqs["condition"] == condition)
                        & (seqs["month"] <= month)]
                idx = [vocab.index(x) for x in df_filt["concept"]]

                item_results = [condition, seq, month]
                col_names = ["condition", "pid", "month"]

                for i, mat in enumerate([G_wd, G_img]):

                    # Min, max, mean distance general
                    item_results += [np.min(np.sum(mat[idx, :], axis=1))/418]
                    item_results += [np.max(np.sum(mat[idx, :], axis=1))/418]
                    item_results += [np.mean(np.sum(mat[idx, :], axis=1))/418]

                    # Min, max, mean distance within subset
                    # Min, max, mean distance general
                    item_results += [np.min(np.sum(mat[idx, :][:, idx], axis=1))/118]
                    item_results += [np.max(np.sum(mat[idx, :][:, idx], axis=1))/118]
                    item_results += [np.mean(np.sum(mat[idx, :][:, idx], axis=1))/118]

                    # Skews (of full distances and subs distances)
                    item_results += [(item_results[4] - item_results[5])/(item_results[4] - item_results[3])]
                    item_results += [(item_results[7] - item_results[8])/(item_results[7] - item_results[6])]


                    col_names = col_names + [str(labs[i]) + x for x in var_names]
                appendage = pd.DataFrame([item_results], columns=col_names)
                results = pd.concat([results, appendage])

    for v in var_names:
        results[v] = results[[x for x in results.columns if v in x]].mean(axis=1)
    results.columns = [
        x
        if x not in ["condition", "pid", "month"]
        else x for x in results.columns]

    
    dist_mets = dist_mets.merge(results, on=["condition", "pid", "month"])
    dist_mets.to_csv(os.path.join(os.path.dirname(currentdir),
                    "results/seq_features/sequence_graph_mets_monthwise_TEST.csv"), mode="a")



def generate_nx_measures(seqs, q=0.1, mode="individual"):
    "For knowledge bases, obtain features which require NetworkX."
    # Load vocab, distance matrices and numpy graphs
    vocab, __, __, G_wd, G_img = utils.load_graphs_and_pw(q=q, dist_type="eu")
    G_img = nx.from_numpy_matrix(G_img, create_using=nx.DiGraph)
    G_wd = nx.from_numpy_matrix(G_wd, create_using=nx.DiGraph)


    conditions = list(seqs["condition"].unique())

    seqs["acq_id"] = [
        i
        for i in range(
            len(seqs.loc[
                (seqs["pid"] == 0) & (seqs["condition"] == "AoA")
        ]))] * (len(conditions) * 200)



    # Load vocab, distance matrices and numpy graphs
    vocab, __, __, G_wd, G_img = utils.load_graphs_and_pw(q=0.1, dist_type="eu")
    G_img = nx.from_numpy_matrix(G_img, create_using=nx.DiGraph)
    G_wd = nx.from_numpy_matrix(G_wd, create_using=nx.DiGraph)

    graph_subs_df = pd.DataFrame({})

    cols = ["condition", "pid", "n_acq", "month", "clustering_full", "clustering_subs", "betweenness_full", "betweenness_subs"]

    header = pd.DataFrame(columns=cols)

    if mode == "monthwise":
        save_fp = os.path.join(os.path.dirname(currentdir),
                    "results/seq_features/sequence_nx_mets_monthwise.csv")
    elif mode == "individual":
        save_fp = os.path.join(os.path.dirname(currentdir),
                    "results/seq_features/sequence_nx_mets_individual.csv")

    header.to_csv(save_fp)


    # Add fullspace distance and degree to df; make into proper df ###########################
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

            # Add fullspace distance and degree to df; make into proper df ###########################
            condition_l = []
            pids = []
            i_s = []
            months = []

            betweenness_subs_l = []
            clustering_subs_l = []
            betweenness_full_l = []
            clustering_full_l = []
            #for s in range(1):

            seq = seqs.loc[(seqs["pid"] == s) & (seqs["condition"] == c)]

            if mode == "individual":
                for i in seq["acq_id"]:
                    print("{}: {}, {}".format(c, s, i))

                    condition_l.append(c)
                    i_s.append(i)
                    pids.append(s)
                    months.append(seq.loc[seq["acq_id"] == i, "month"].iloc[0])

                    curr = vocab.index(seq.loc[seq["acq_id"] == i, "concept"].iloc[0])

                    img_clustering_full = nx.clustering(G_img, nodes=[curr])[curr]
                    wd_clustering_full = nx.clustering(G_wd, nodes=[curr])[curr]
                    clustering_full = (img_clustering_full + wd_clustering_full)/2

                    wd_betweenness_full = nx.betweenness_centrality(G_wd)[curr]
                    img_betweenness_full = nx.betweenness_centrality(G_img)[curr]
                    betweenness_full = (img_betweenness_full + wd_betweenness_full)/2
                    
                    if i > 0:
                        seq_subs = seq.loc[seq["acq_id"] <= i]
                        idx = [vocab.index(x) for x in seq_subs["concept"]]
                        subs_vocab = [vocab[x] for x in idx]

                        G_wd_subs = G_wd.subgraph(idx)
                        G_img_subs = G_img.subgraph(idx)

                        img_clustering_subs = nx.clustering(G_img_subs, nodes=[curr])[curr]
                        wd_clustering_subs = nx.clustering(G_wd_subs, nodes=[curr])[curr]
                        clustering_subs = (img_clustering_subs + wd_clustering_subs)/2

                        wd_betweenness_subs = nx.betweenness_centrality(G_wd_subs)[curr]
                        img_betweenness_subs = nx.betweenness_centrality(G_img_subs)[curr]
                        betweenness_subs = (img_betweenness_subs + wd_betweenness_subs)/2
                        
                    else:
                        clustering_subs = 0
                        betweenness_subs = 0
                    
                    clustering_full_l.append(clustering_full)
                    clustering_subs_l.append(clustering_subs)
                    betweenness_full_l.append(betweenness_full)
                    betweenness_subs_l.append(betweenness_subs)
    
            elif mode == "monthwise":
                for m in list(set(seq["month"])):
                    print(m)

                    condition_l.append(c)
                    i_s.append(0)
                    pids.append(s)
                    months.append(m)

                    # Get mean betweenness and clustering for all items in sample in full space
                    curr = [vocab.index(x) for x in list(seq.loc[seq["month"] <= m, "concept"])]

                    img_clustering_full = np.mean([nx.clustering(G_img, nodes=curr)[x] for x in curr])
                    wd_clustering_full = np.mean([nx.clustering(G_wd, nodes=curr)[x] for x in curr])
                    clustering_full = (img_clustering_full + wd_clustering_full)/2

                    nx_bw_wd = nx.betweenness_centrality(G_wd)
                    nx_bw_img = nx.betweenness_centrality(G_img)
                    wd_betweenness_full = np.mean([nx_bw_wd[x] for x in curr])
                    img_betweenness_full = np.mean([nx_bw_img[x] for x in curr])
                    betweenness_full = (img_betweenness_full + wd_betweenness_full)/2
                    
                    subs_vocab = [vocab[x] for x in curr]

                    G_wd_subs = G_wd.subgraph(curr)
                    G_img_subs = G_img.subgraph(curr)

                    img_clustering_subs = np.mean([nx.clustering(G_img_subs, nodes=curr)[x] for x in curr])
                    wd_clustering_subs =np.mean([nx.clustering(G_wd_subs, nodes=curr)[x] for x in curr])
                    clustering_subs = (img_clustering_subs + wd_clustering_subs)/2

                    nx_bw_wd = nx.betweenness_centrality(G_wd_subs)
                    nx_bw_img = nx.betweenness_centrality(G_img_subs)
                    wd_betweenness_subs = np.mean([nx_bw_wd[x] for x in curr])
                    img_betweenness_subs = np.mean([nx_bw_img[x] for x in curr])
                    betweenness_subs = (img_betweenness_subs + wd_betweenness_subs)/2
                        

                    # get mean betweenness and clustering for items in sample within sample

                    clustering_full_l.append(clustering_full)
                    clustering_subs_l.append(clustering_subs)
                    betweenness_full_l.append(betweenness_full)
                    betweenness_subs_l.append(betweenness_subs)



            df = pd.DataFrame({
                "condition": condition_l,
                "pid":pids,
                "n_acq": i_s,
                "month": months,
                "clustering_full": clustering_full_l,
                "clustering_subs": clustering_subs_l,
                "betweenness_full": betweenness_full_l,
                "betweenness_subs": betweenness_subs_l,
            })

            df.to_csv(save_fp, mode="a", header=None)

     
def get_coverage(seqs, mode="individual"):

    # Quickload
    wdnimg = np.loadtxt(os.path.join(
                            os.path.dirname(currentdir),
                            "assets/embeddings/wdnimg.txt"))
    imgnwd = np.loadtxt(os.path.join(
                            os.path.dirname(currentdir),
                            "assets/embeddings/imgnwd.txt"))


    # Get coverage range in each dimension
    img_full_rng = np.max(imgnwd, axis=0) - np.min(imgnwd, axis=0)
    wd_full_rng = np.max(wdnimg, axis=0) - np.min(wdnimg, axis=0)

    textfile = open(os.path.join(
                            os.path.dirname(currentdir),
                            "assets/embeddings/vocab.txt"), "r")
    vocab = textfile.read().split('\n')

    conditions = list(seqs["condition"].unique())

    seqs["acq_id"] = [
        i
        for i in range(
            len(seqs.loc[
                (seqs["pid"] == 0) & (seqs["condition"] == "AoA")
        ]))] * (len(conditions) * 200)

    #seqs = seqs.loc[seqs["condition"] == "control"]
    # Add fullspace distance and degree to df; make into proper df ###########################

    cols = ["condition", "pid", "n_acq", "month", "coverage_wd", "coverage_img", "coverage"]

    header = pd.DataFrame(columns=cols)

    if mode == "individual":
        save_fp = os.path.join(os.path.dirname(currentdir),
                    "results/sequence_item_coverage.csv")

    elif mode == "monthwise":
        save_fp = os.path.join(os.path.dirname(currentdir),
                    "results/seq_features/sequence_coverage_monthwise_TEST.csv")
    
    #if not os.path.exists(save_fp):
        #print("ji")
    header.to_csv(save_fp)

    for c in conditions:
        for s in range(100):

            condition_l = []
            pids = []
            i_s = []
            months = []
            
            coverages_wd = []
            coverages_img = []
            coverages = []

            seq = seqs.loc[(seqs["pid"] == s) & (seqs["condition"] == c)]
            

            if mode == "individual":
                for i in seq["acq_id"]:
                    condition_l.append(c)
                    i_s.append(i)
                    pids.append(s)
                    months.append(seq.loc[seq["acq_id"] == i, "month"].iloc[0])

                    if i > 0:
                        subs_acq = seq.loc[seq["acq_id"] < i, "concept"]
                        subs_idx = [vocab.index(x) for x in list(subs_acq)]
                        
                        # Get sample coverage 
                        img_subs_rng = np.max(imgnwd[subs_idx, :], axis=0) - np.min(imgnwd[subs_idx, :], axis=0)
                        wd_subs_rng = np.max(wdnimg[subs_idx, :], axis=0) - np.min(wdnimg[subs_idx, :], axis=0)

                        # Get proportion
                        img_coverage = np.mean(np.divide(img_subs_rng, img_full_rng))
                        wd_coverage = np.mean(np.divide(wd_subs_rng, wd_full_rng))
                        coverage = (wd_coverage + img_coverage)/2
                        
                        coverages_wd.append(wd_coverage)
                        coverages_img.append(img_coverage)
                        coverages.append(coverage)
                        

                    else:
                        coverages_wd.append(0)
                        coverages_img.append(0)
                        coverages.append(0)
            
            else:
                for m in seq["month"].unique():

                    condition_l.append(c)
                    i_s.append(0)
                    pids.append(s)
                    months.append(m)

                    subs_idx = [vocab.index(x) for x in list(seq.loc[seq["month"] <= m, "concept"])]

                    # Get sample coverage 
                    img_subs_rng = np.max(imgnwd[subs_idx, :], axis=0) - np.min(imgnwd[subs_idx, :], axis=0)
                    wd_subs_rng = np.max(wdnimg[subs_idx, :], axis=0) - np.min(wdnimg[subs_idx, :], axis=0)

                    # Get proportion
                    img_coverage = np.mean(np.divide(img_subs_rng, img_full_rng))
                    wd_coverage = np.mean(np.divide(wd_subs_rng, wd_full_rng))
                    coverage = (wd_coverage + img_coverage)/2
                    
                    coverages_wd.append(wd_coverage)
                    coverages_img.append(img_coverage)
                    coverages.append(coverage)
                        


            df = pd.DataFrame({
                "condition": condition_l,
                "pid":pids,
                "n_acq": i_s,
                "month": months,
                "coverages_wd": coverages_wd,
                "coverages_img": coverages_img,
                "coverages": coverages
            })

            df.to_csv(save_fp, mode="a", header=None)


if __name__ == "__main__":

    aoa_seqs = pd.read_csv(
            os.path.join(parentdir, "results/sample_sequences/AoA_sequences.csv"),
            header=0, index_col=0
        )

    control_seqs_exc = pd.read_csv(
            os.path.join(parentdir, "results/sample_sequences/controlExcAoA_sequences.csv"),
            header=0, index_col=0
    )

    
    control_seqs_inc = pd.read_csv(
        os.path.join(parentdir, "results/sample_sequences/controlIncAoA_sequences.csv"),
            header=0, index_col=0
    )

    seqs = pd.concat([aoa_seqs, control_seqs_inc, control_seqs_exc])
    ##### CONSISTENT TO HERE 0

    generate_non_nx_measures(seqs)
    get_coverage(seqs, mode="monthwise")

    #generate_nx_measures(seqs, mode="monthwise")

    """
    df = pd.read_csv(os.path.join(os.path.dirname(currentdir),
                    "results/sequence_item_mets_monthwise.csv"))

    print(df)

    print(os.path.join(os.path.dirname(currentdir),
                    "results/sequence_item_mets_monthwise.csv"))
    """
