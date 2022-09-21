from re import L
import sys
from tkinter import S

from numpy.core.fromnumeric import ptp
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
import tensorflow as tf
import numpy as np
import pandas as pd
import networkx as nx

import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "" # "1"


class FeatureModel:
    "Parent class for models which optimise structural features."
    def __init__(self, months, restart, 
                var_list, save_fp, sigmoid, dist_type
                ):

        self.dist_type=dist_type
        vocab, wd_pw, img_pw, G_wd, G_img = utils.load_graphs_and_pw(q=0.1,
                                                    dist_type=dist_type)

        self.vocab = vocab
        self.wd_pw_np = wd_pw
        self.img_pw_np = img_pw
        self.G_wd_np = G_wd
        self.G_img_np = G_img
        self.wd_pw = tf.constant(wd_pw, dtype="float32")
        self.img_pw = tf.constant(img_pw, dtype="float32")
        self.G_img = tf.constant(G_img, dtype="float32")
        self.G_wd = tf.constant(G_wd, dtype="float32")
        self.restart = restart
        self.var_list = var_list
        self.mod_type=None #placeholder

        # Load reference data for AoA, concepts by month etc
        aoa_dat = utils.load_aoa_data() # Remove reliance on neighbourhood overlap, given we're not filtering
        aoa_dat = aoa_dat.loc[aoa_dat["concept_i"].isin(vocab)]
        self.aoa_dat = aoa_dat

        # Load concepts acquired by month
        ___, mean_concepts_acquired = utils.monthly_concepts(aoa_dat)
        if months is None:
            months = [x for x in list(mean_concepts_acquired.keys())]
        else:
            mean_concepts_acquired = {k:mean_concepts_acquired[k]
                                      for k in months} 
        self.months = months
        self.mean_concepts_acquired = mean_concepts_acquired

        # Partitioning, if any
        self.candidate_idxs = [x for x in range(len(vocab))] # Partitioning : candidate set

        # Quickload
        wdnimg = np.loadtxt(
            os.path.join(parentdir, "assets/embeddings/wdnimg.txt")
        )
        imgnwd = np.loadtxt(
            os.path.join(parentdir, "assets/embeddings/imgnwd.txt")
        )
        self.imgnwd = imgnwd
        self.wdnimg = wdnimg

        # Get coverage range in each dimension
        img_full_rng = np.max(imgnwd, axis=0) - np.min(imgnwd, axis=0)
        wd_full_rng = np.max(wdnimg, axis=0) - np.min(wdnimg, axis=0)
        self.img_full_rng = img_full_rng
        self.wd_full_rng = wd_full_rng


        self.sigmoid = sigmoid
        self.save_fp = save_fp


    def loss():
        "Loss function placeholder."
        raise NotImplementedError("Subclass needs to define this.")


    def val_loss():
        "Validation loss placeholder."
        raise NotImplementedError("Subclass needs to define this.")


    def _init_file(self):
        "Initialise file to record training progress."
        cols = ["loss", "month", "month_n", "epoch"]
        cols = cols + self.var_list + ["linear_comb", "val_loss", "restart"]

        header = pd.DataFrame(columns=cols)
        if not os.path.exists(self.save_fp):
            header.to_csv(self.save_fp)
        self.cols = cols


    def softmax(self, logits, temp=1):
        "Softmax function to convert distribution to probability dist."
        temp_logits = tf.divide(logits, tf.cast(temp, dtype="float32"))
        bottom = tf.math.exp(temp_logits)
        softmax = tf.divide(
                            tf.math.exp(temp_logits),
                            tf.reduce_sum(bottom, axis=-1, keepdims=True))
        return softmax


    def get_remaining_concepts(self, acquired_concepts):
        "Given acquired concept list, retrieve remaining concepts."
        if acquired_concepts is not None:
            remaining_concepts = [
                x for x in self.candidate_idxs
                if x not in list(acquired_concepts.numpy())]
        else:
            remaining_concepts = [x for x in self.candidate_idxs]
        
        return remaining_concepts


    def _get_candidate_coverage(self, acquired_concepts):
        "Get average dimension coverage of candidate knowledge bases."

        remaining_concepts = self.get_remaining_concepts(acquired_concepts)

        if acquired_concepts is None:

            cand_coverages = [0 for x in range(len(remaining_concepts))]

        else:

            cand_coverages = []

            for c in remaining_concepts:
                subs_idx = [int(x) for x in list(acquired_concepts.numpy())] + [c]
                
                # Get coverage established by each candidate item
                img_subs_rng = (
                    np.max(self.imgnwd[subs_idx, :], axis=0)
                    - np.min(self.imgnwd[subs_idx, :], axis=0)
                )
                wd_subs_rng = (
                    np.max(self.wdnimg[subs_idx, :], axis=0)
                    - np.min(self.wdnimg[subs_idx, :], axis=0)
                )

                # Get proportion
                img_coverage = np.mean(np.divide(img_subs_rng, self.img_full_rng))
                wd_coverage = np.mean(np.divide(wd_subs_rng, self.wd_full_rng))
                co = (wd_coverage + img_coverage)/2
                cand_coverages.append(co)
        
        return cand_coverages


    def _get_candidate_clustering_subs(self, acquired_concepts):
        "Get clustering within candidate knowledge bases."
        
        remaining_concepts = self.get_remaining_concepts(acquired_concepts)
        if acquired_concepts is None:
             cand_clustering = [0 for x in range(len(self.candidate_idxs))]

        else:
            cand_clustering = []

            for c in remaining_concepts:
                subs_idx = [int(x)
                            for x in list(acquired_concepts.numpy())] + [c]
                
                # Get clustering established by each candidate item
                gi_nx = nx.from_numpy_matrix(
                    self.G_img_np[subs_idx,:][:, subs_idx]
                    )
                ci = np.mean(list(nx.clustering(gi_nx).values()))

                gw_nx = nx.from_numpy_matrix(
                    self.G_wd_np[subs_idx,:][:, subs_idx])
                cw = np.mean(list(nx.clustering(gw_nx).values()))
                
                c = (ci + cw)/2
                cand_clustering.append(c)
        
        return cand_clustering


    def _get_candidate_degree_subs(self, acquired_concepts):
        "Get mean degree of candidate concepts within knowledge base."

        remaining_concepts = self.get_remaining_concepts(acquired_concepts)

        if acquired_concepts is None:
            subs_deg = np.zeros(len(self.candidate_idxs))

        else:

            subs_idx = np.array(list(acquired_concepts.numpy())).astype("int")

            subs_deg_wd = np.sum(
                self.G_wd_np[remaining_concepts, :][:, subs_idx], axis=1)
            subs_deg_img = np.sum(
                self.G_img_np[remaining_concepts, :][:, subs_idx], axis=1)

            subs_deg = (subs_deg_wd + subs_deg_img)/2

            # Scale to be between 0 and 1
            subs_deg = subs_deg/np.max(
                np.array(list(self.mean_concepts_acquired.values()))
                )

        return subs_deg


    def _get_candidate_degree_full(self, acquired_concepts):
        "Get degrees of candidate concepts in full concept space."

        remaining_concepts = self.get_remaining_concepts(acquired_concepts)

        full_deg_wd = np.sum(self.G_wd_np[remaining_concepts, :], axis=1)
        full_deg_img = np.sum(self.G_img_np[remaining_concepts, :], axis=1)

        full_deg = (full_deg_wd + full_deg_img)/2
        full_deg = full_deg/len(self.vocab)

        return full_deg


    def _get_candidate_max_dist_subs(self, acquired_concepts):
        "Get max dist of candidate concepts from kb concepts."
        
        remaining_concepts = self.get_remaining_concepts(acquired_concepts)

        if acquired_concepts is None:
            subs_dist = np.zeros(len(self.candidate_idxs))

        else:
            subs_idx = np.array(list(acquired_concepts.numpy())).astype("int")

            subs_dist_wd = np.max(
                self.wd_pw_np[remaining_concepts, :][:, subs_idx], axis=1)
            subs_dist_img = np.max(
                self.img_pw_np[remaining_concepts, :][:, subs_idx], axis=1)

            subs_dist = (subs_dist_wd + subs_dist_img)/2

        return subs_dist


    def _get_candidate_mean_dist_subs(self, acquired_concepts):
        "Get mean dist of candidate concepts from  kb concepts."

        remaining_concepts = self.get_remaining_concepts(acquired_concepts)

        if acquired_concepts is None:
            subs_dist = np.zeros(len(self.candidate_idxs))

        else:
            subs_idx = np.array(list(acquired_concepts.numpy())).astype("int")

            subs_dist_wd = np.mean(
                self.wd_pw_np[remaining_concepts, :][:, subs_idx], axis=1)
            subs_dist_img = np.mean(
                self.img_pw_np[remaining_concepts, :][:, subs_idx], axis=1)

            subs_dist = (subs_dist_wd + subs_dist_img)/2

        return subs_dist

    
    def _get_candidate_min_dist_subs(self, acquired_concepts):
        "Get min dist of candidate concepts from  kb concepts."
        
        remaining_concepts = self.get_remaining_concepts(acquired_concepts)

        if acquired_concepts is None:
            subs_dist = np.zeros(len(self.candidate_idxs))

        else:
            subs_idx = np.array(list(acquired_concepts.numpy())).astype("int")

            subs_dist_wd = np.min(
                self.wd_pw_np[remaining_concepts, :][:, subs_idx], axis=1)
            subs_dist_img = np.min(
                self.img_pw_np[remaining_concepts, :][:, subs_idx], axis=1)

            subs_dist = (subs_dist_wd + subs_dist_img)/2

        return subs_dist


    def _get_candidate_max_dist_full(self, acquired_concepts):
        "Get max dist of candidate concepts from all concepts."

        remaining_concepts = self.get_remaining_concepts(acquired_concepts)

        full_dist_wd = np.max(self.wd_pw_np[remaining_concepts, :], axis=1)
        full_dist_img = np.max(self.img_pw_np[remaining_concepts, :], axis=1)

        subs_dist = (full_dist_wd + full_dist_img)/2

        return subs_dist


    def _get_candidate_min_dist_full(self, acquired_concepts):
        "Get mean dist of candidate concepts from all concepts."

        remaining_concepts = self.get_remaining_concepts(acquired_concepts)

        full_dist_wd = np.min(self.wd_pw_np[remaining_concepts, :], axis=1)
        full_dist_img = np.min(self.img_pw_np[remaining_concepts, :], axis=1)

        subs_dist = (full_dist_wd + full_dist_img)/2

        return subs_dist


    def _get_candidate_mean_dist_full(self, acquired_concepts):
        "Get mean dist of candidate concepts from all concepts."
        
        remaining_concepts = self.get_remaining_concepts(acquired_concepts)

        full_dist_wd = np.min(self.wd_pw_np[remaining_concepts, :], axis=1)
        full_dist_img = np.min(self.img_pw_np[remaining_concepts, :], axis=1)

        subs_dist = (full_dist_wd + full_dist_img)/2

        return subs_dist


    def _get_candidate_degree_skew_full(self, acquired_concepts):
        "Get skew of candidate kbs' degrees in full space."

        remaining_concepts = self.get_remaining_concepts(acquired_concepts)

        if acquired_concepts is None:
            skews = np.zeros(len(self.candidate_idxs))

        else:
            skews=[]
            subs_idx = np.array(
                [int(x) for x in list(acquired_concepts.numpy())]
                )
            subs_idx = np.tile(subs_idx, (len(remaining_concepts), 1))
            rem = np.array(remaining_concepts).reshape(
                                            len(remaining_concepts), 1
                                            )
            subs_idx = np.append(subs_idx, rem, axis=1)

            # Get degrees for subsets
            wd_degs_all = np.squeeze(np.sum(self.G_wd_np, axis=1))
            img_degs_all = np.squeeze(np.sum(self.G_img_np, axis=1))

            wd_degs = np.take(wd_degs_all, subs_idx)
            img_degs = np.take(img_degs_all, subs_idx)

            min_deg_wd = np.min(wd_degs, axis=1)/len(self.vocab)
            min_deg_img = np.min(img_degs, axis=1)/len(self.vocab)
            
            mean_deg_wd = np.mean(wd_degs, axis=1)/len(self.vocab)
            mean_deg_img = np.mean(img_degs, axis=1)/len(self.vocab)

            max_deg_wd = np.max(wd_degs, axis=1)/len(self.vocab)
            max_deg_img = np.max(img_degs, axis=1)/len(self.vocab)

            img_skew = np.nan_to_num(
                        (max_deg_img-mean_deg_img)/(max_deg_img-min_deg_img)
                        ) # Skew is between 0 and 1, 0.5 is no skew, 0 is max neg skew, 1 is max pos skew
            wd_skew = np.nan_to_num(
                        (max_deg_wd-mean_deg_wd)/(max_deg_wd-min_deg_wd)
                        )

            skews.append((img_skew + wd_skew)/2)
            skews = np.array(skews)

        return skews


    def _get_candidate_degree_skew_subs(self, acquired_concepts):
        "Get skew of candidate kbs' degrees in kb."

        remaining_concepts = self.get_remaining_concepts(acquired_concepts)

        if acquired_concepts is None:
            skews = np.zeros(len(self.candidate_idxs))

        else:
            skews=[]

            for x in remaining_concepts:
                subs_idx = np.array(
                    [int(x) for x in list(acquired_concepts.numpy())] + [x]
                    )

                wd_degs = np.squeeze(
                    np.sum(self.G_wd_np[subs_idx, :][:,subs_idx], axis=1)
                    )
                img_degs = np.squeeze(
                    np.sum(self.G_img_np[subs_idx, :][:,subs_idx], axis=1)
                    )

                min_deg_wd = np.min(wd_degs)/len(self.vocab)
                min_deg_img = np.min(img_degs)/len(self.vocab)
                
                mean_deg_wd = np.mean(wd_degs)/len(self.vocab)
                mean_deg_img = np.mean(img_degs)/len(self.vocab)

                max_deg_wd = np.max(wd_degs)/len(self.vocab)
                max_deg_img = np.max(img_degs)/len(self.vocab)

                img_skew = np.nan_to_num(
                    (max_deg_img-mean_deg_img)/(max_deg_img-min_deg_img)
                    ) # Skew is between 0 and 1, 0.5 is no skew, 0 is max neg skew, 1 is max pos skew
                wd_skew = np.nan_to_num(
                    (max_deg_wd-mean_deg_wd)/(max_deg_wd-min_deg_wd)
                    )

                skews.append((img_skew + wd_skew)/2)
            skews = np.array(skews)

        return skews


    def get_probs(self, all_vars, cand_vals):
        "Get probability distribution across candidate concepts."

        targets = {}
        scores = []
        for v in self.var_list:
            v_t = all_vars[v]
            target = tf.expand_dims(
                            tf.cast(tf.expand_dims(v_t, 0), 'float32'), -1
                            )
            
            # Score is difference from target value
            score = -tf.math.sqrt(tf.math.maximum(
                                        tf.math.square(
                                                tf.math.subtract(
                                                    cand_vals[v],
                                                    tf.multiply(
                                                        tf.ones_like(
                                                            cand_vals[v],
                                                            dtype="float32"
                                                            ),
                                                        target))),
                                                    tf.constant([1e-18])))
            scores.append(score)
        score_mat = tf.transpose(tf.convert_to_tensor(
                                    scores,
                                dtype="float32"), [1,0,2])

        score = tf.linalg.matmul(tf.math.exp(all_vars["linear_comb"]),  ## Constrain weights to be positive
                                        score_mat)
        
        # Rescale
        score = score - tf.reduce_min(score, axis=-1, keepdims=True)
        score = score / (
                        tf.reduce_max(score, axis=-1, keepdims=True)
                        + tf.constant([1e-18])
                        )

        probs = self.softmax(score, temp=5e-2)

        return probs


    def get_next_item(self, all_vars, cand_vals, acquired_concepts,
                      remaining_concepts):
        "Select next candidate to be added to knowledge base."

        if self.sigmoid:
            sig_vars = { k:(tf.sigmoid(v) if k!= "linear_comb" else v)
                         for k, v in all_vars.items()
                        }
        else:
            sig_vars = all_vars
        
        probs = self.get_probs(sig_vars, cand_vals)

        remaining_concepts = tf.constant(remaining_concepts, dtype="float32")

        idx = tf.cast(tf.squeeze(
            tf.random.categorical(tf.squeeze(tf.math.log(probs), axis=0), 1)),
            "int32")

        addn = tf.expand_dims(tf.gather(remaining_concepts, idx), 0)

        if acquired_concepts is not None:
            return tf.concat([acquired_concepts, addn], axis=0)
        else:
            return addn


    def train_cc(self, n_epochs):
        "Train to seek appropriate target values for selected variables."

        # Init tracking file
        self._init_file()

        linear_comb = tf.Variable(
                                tf.ones((1,len(self.var_list))),
                                dtype="float32", name="linear_comb")

        cand_vals = {}
        all_vars = {
            "linear_comb":linear_comb
        }

        for v in self.var_list:
            
            if self.sigmoid:
                all_vars[v] = tf.Variable(tf.squeeze(tf.random.uniform(
                                            [1], minval=-2, maxval=2 #, seed=(restart*2)*seed_int + 4
                                            )),
                                        name=v, dtype="float32")
            else: 
                all_vars[v] = tf.Variable(tf.squeeze(tf.random.uniform(
                            [1], minval=0, maxval=1 #, seed=(restart*2)*seed_int + 4
                            )),
                        name=v, dtype="float32")

        # Commence training:
        for epoch in range(n_epochs):
            acquired_concepts = None
            val_loss = 0

            for month in self.months:
                self.month = month
                n_acq = self.mean_concepts_acquired[month]

                for i in range(n_acq):

                    # Get candidate var values for vars of interest
                    for v in self.var_list:
                        function_name = "_get_candidate_" + v
                        args = [acquired_concepts]

                        cand_vals[v] = getattr(self, function_name)(*args)

                    remaining_concepts = self.get_remaining_concepts(
                                                        acquired_concepts
                                                        )
                    acquired_concepts = self.get_next_item(
                                                        all_vars, cand_vals,
                                                        acquired_concepts,
                                                        remaining_concepts)

                    if i == n_acq-2:

                        # Get candidate var values for vars of interest
                        for v in self.var_list:
                            function_name = "_get_candidate_" + v
                            args = [acquired_concepts]

                            cand_vals[v] = getattr(self, function_name)(*args)  

                        with tf.GradientTape(persistent=True) as tape:

                            remaining_concepts = self.get_remaining_concepts(
                                acquired_concepts
                                )
                            loss = self.loss(
                                            all_vars,
                                            cand_vals,
                                            acquired_concepts,
                                            remaining_concepts,
                                            mode="train")

                        grads = tape.gradient(loss, list(all_vars.values()))

                        self.optimizer.apply_gradients(
                                            zip(grads, list(all_vars.values()))
                                            )

                        val_loss = self.loss(
                                            all_vars,
                                            cand_vals,
                                            acquired_concepts,
                                            remaining_concepts,
                                            mode="val")

                        df_dict = {
                                "loss": [loss.numpy()],
                                "month": [month],
                                "month_n": [i],
                                "epoch": [epoch],
                                "linear_comb": [linear_comb.numpy()],
                                "val_loss": [val_loss.numpy()],
                                "restart":[self.restart]
                        }

                        for v in self.var_list:
                            df_dict[v] = [all_vars[v].numpy()]

                        df = pd.DataFrame(df_dict)
                        # Rearrange columns
                        df = df[self.cols]
                        df.to_csv(self.save_fp.format(
                                            self.mod_type, self.restart),
                                  mode="a", header=None)

                        print(
                            "Epoch: {}, Month: {}, Loss: {}, n: {}".format(
                                epoch, month, loss, i)
                            )


class ProbMatch(FeatureModel):
    "Model aiming to maximise prob dist proximity to AoA data."
    def __init__(self, months=None, restart=0, 
                var_list=["degree_full"],
                save_fp=os.path.join(
                            parentdir,
                            "results/run_2/model_training/{}_results_{}.csv"),
                train_split=0.7, sigmoid=True, dist_type="eu"):
        super().__init__(months, restart, var_list, save_fp, sigmoid,
                         dist_type)

        # Load in bootstrapped probabilities for train/test probs
        bs_dists = pd.read_csv(
            os.path.join(
                    parentdir,
                    "assets/bootstrap_dists/bootstrap_distributions.csv"
                ),
                header=0, index_col=0
            )
        self.bs_dists = bs_dists
        n_bs = bs_dists["bootstrap"].max()
        self.n_bs = n_bs

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)

        # Split into train and val sets
        self.train_split = train_split
        train = bs_dists.loc[bs_dists["bootstrap"] < n_bs * train_split]
        val = bs_dists.loc[bs_dists["bootstrap"] >= n_bs * train_split]
        self.train = train
        self.val = val
        self.mod_type="probMatch"

        self.save_fp = self.save_fp.format(self.mod_type, self.restart)


    def get_month_stats(self):
        """Load reference mean/std MSE loss values sampled by month."""
        sample_means = pd.read_csv(
            os.path.join(
                        parentdir,
                        "assets/sample_mse_aoa/sample_summaries.csv"),
                        header=0)

        month_mean = sample_means.loc[
            sample_means["month"] == int(self.month), "mean"
            ].iloc[0]
        month_std = sample_means.loc[
            sample_means["month"] == int(self.month), "std"
            ].iloc[0]

        return month_mean, month_std


    def get_bootstrapped_pcs(self, df, concepts):
        """Transform bootstrapped distributions into format for y_true."""
        
        # Get validation set       
        y = df.loc[
                df["month"] == int(self.month),
                ["bootstrap", "concept", "cumsum"]
            ]
        y = y.pivot_table(index="bootstrap", columns=["concept"])
        y = y.droplevel(0,1)
        y[[x for x in concepts if x not in y.columns]] = 0
        # Reorder columns to align with vocab filt
        y = y[concepts]

        y = np.array(y)
        y = np.mean(y, axis=0)
        return y


    def loss(self, all_vars, cand_vals, acquired_concepts, remaining_concepts,
            mode="train"):
        """Calculate loss probMatch loss."""

        if mode == "train":
            df = self.train
        elif mode == "val":
            df = self.val

        if self.sigmoid:
            sig_vars = { k:(tf.sigmoid(v) if k!= "linear_comb" else v)
                         for k, v in all_vars.items()
                        }
        else:
            sig_vars = all_vars

        probs = self.get_probs(sig_vars, cand_vals)

        # Probs for acquired items
        out = tf.ones(acquired_concepts.shape[0])
        
        # Followed by probs for remaining
        out = tf.concat([out, tf.squeeze(probs)], axis=0)

        # Make sure y_true aligns with these concepts
        concept_order = list(
                                acquired_concepts.numpy().astype("int")
                            ) + remaining_concepts
        concept_names = [self.vocab[x] for x in concept_order]
        y_true = self.get_bootstrapped_pcs(df, concept_names)
        y_true = tf.constant(y_true)

        loss_func = tf.keras.losses.MeanSquaredError()
        #mult = tf.constant([y_true.shape[0]])
        #out = tf.tile(out, mult)
        loss = loss_func(out, y_true)

        month_mean, month_std = self.get_month_stats()

        loss = (loss - month_mean)/month_std
        min_loss = (tf.constant(0, dtype="float64") - month_mean)/month_std  # 0 = theoretical minimum value of loss 
        loss = loss - min_loss

        return loss

        
class Optimal(FeatureModel):
    """Model trained to optimise forced choice performance."""
    def __init__(self, 
                months=None, restart=0,
                var_list=["degree_full"],
                save_fp=os.path.join(
                            parentdir,
                            "results/run_2/model_training/{}_results_{}.csv"),
                sigmoid=True, dist_type="eu"):
        super().__init__(months, restart, var_list, save_fp, sigmoid,
                         dist_type)

        self.sample_means = pd.read_csv(
            os.path.join(
                        parentdir,
                        "assets/sample_pair_alignments/sample_summaries.csv"),
            header=0)
        self.mod_type = "optimal"
        self.save_fp = self.save_fp.format(self.mod_type, self.restart)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


    def get_partitions(self, n_concepts, aoa_idx,
                        n_required=25, acquired_concepts=None,
                        split={"candidate":100, "train":100, "val":100, "test":100}):
        """
        Split concept indices into partitions required for generative experiment
        :: n_concepts :: total number of concepts to chooose from
        :: aoa_idx :: list of indices of items which are early acquired
        :: n_required :: number of items in eventual knowledge base 
        :: (== minimum number of AoA concepts which can be in candidate set)
        """

        acquired_concepts = list(acquired_concepts.numpy())

        if sum(split.values()) > n_concepts:
            raise ValueError(
                "Total number of items in splits must be < number of concepts"
                )

        idxs = [x for x in range(n_concepts)]
        n_acquired=0
        acq_idxs = []


        if acquired_concepts is not None:
            acq_idxs = acquired_concepts
            n_acquired = len(acquired_concepts)
            idxs = [x for x in range(n_concepts) if x not in acq_idxs]

        n_aoa = 0
        train_idxs = []
        val_idxs = []
        test_idxs = []

        candidate_idxs = list(np.random.choice(
                        idxs, split["candidate"] - n_acquired, replace=False
                        ))

        candidate_idxs = candidate_idxs + [int(x) for x in acquired_concepts]

        while n_aoa < n_required:

            # Check that number of AoA concepts meets required threshole to generate AoA set
            candidate_idxs = list(np.random.choice(
                idxs, split["candidate"], replace=False
                ))
            n_aoa = sum([x in aoa_idx for x in candidate_idxs])
        
        if split["train"] > 0:
            train_idxs = list(np.random.choice(
                [x for x in idxs if x not in candidate_idxs],
                split["train"], replace=False
                ))
        if split["val"] > 0:
            val_idxs = list(np.random.choice(
                [x for x in idxs if x not in train_idxs + candidate_idxs],
                split["val"], replace=False)
            )
        if split["test"] > 0:
            test_idxs =  list(np.random.choice(
                [x for x in idxs
                if x not in train_idxs + val_idxs + candidate_idxs],
                split["test"], replace=False))


        return candidate_idxs, train_idxs, val_idxs, test_idxs


    def get_probe_pair_set(self, set_idxs, n_pairs):
        """
        Generate probe pairs for optimal teaching test
        """
    
        pair_df = pd.DataFrame(columns=["item_i", "item_j", "idx_i", "idx_j"])
        all_tried = []
        while len(pair_df) < n_pairs:
            test_idxs = np.random.choice(
                set_idxs, 2, replace=False
            )
            # Check if pair has been added in either order before
            sorted_idx = test_idxs.copy()
            sorted_idx.sort()

            if tuple(sorted_idx) not in all_tried:
                append = pd.DataFrame({
                    "item_i": [self.vocab[test_idxs[0]]],
                    "item_j": [self.vocab[test_idxs[1]]],
                    "idx_i": [test_idxs[0]],
                    "idx_j": [test_idxs[1]]
                })
                pair_df = pd.concat([pair_df, append])
                all_tried.append(tuple(sorted_idx))

        return pair_df


    def prob_matrix(self, probs, acquired_concepts, remaining_concepts):
        """Ensure that order of concepts is acquired then remaining)"""

        probs = tf.squeeze(probs)

        if acquired_concepts is None:
            raise NotImplementedError(
                "No alignment loss when no concepts have been acquired."
                )

        else:
            # Probs for pre-acquired knowledge
            out = tf.ones(acquired_concepts.shape[0])

        # Probs for newly sampled items
        out = tf.concat([out, probs], axis=0)

        # Append 2 weights of 1 for test items
        out = tf.concat([out, tf.ones((2,))], axis=0)

        # Get probability matrix across all concepts
        pm = tf.matmul(tf.expand_dims(out,1), tf.expand_dims(out, 0))

        if acquired_concepts is not None:
            # Set all values where both i and j are newly acquired to zero
            # Get TFTFTF matrix for if index in indices or not
            acquired_bool = tf.ones(
                    (tf.squeeze(acquired_concepts).shape[0], 1)
                    )
            acquired_bool = tf.concat(
                    [acquired_bool, tf.ones((len(remaining_concepts), 1))],
                    axis=0)

            # Tile
            acquired_tile = tf.tile(
                tf.concat([acquired_bool, tf.ones((2,1))],
                          axis=0), tf.constant([1, pm.shape[0]]))

            # Logical any (if any is acquired, keep the relationship, else set to 0)
            any_acq = tf.reduce_any(
                tf.stack([tf.cast(acquired_tile, dtype="bool"),
                          tf.cast(tf.transpose(acquired_tile), dtype="bool")],
                        axis=0), axis=0
                )

            pm = tf.multiply(pm, tf.cast(any_acq, dtype="float32"))

        return pm


    def _get_alignment_loss(self, concept_order, pair_df, pm):
        """Calculate mean alignment loss across concept pairs."""
        
        tot_loss = 0

        concept_order = tf.constant(concept_order, dtype="int32")

        for index, row in pair_df.iterrows():
            test_idx = [row["idx_i"], row["idx_j"]]
            correct_map = tf.concat(
                [concept_order, tf.constant(test_idx, dtype="int32")], axis=0)
            incorrect_map =  tf.concat([concept_order, tf.constant([
                test_idx[1], test_idx[0]
                ], dtype="int32")], axis=0)    

            wd_pw_wt = tf.gather(
                                tf.gather(self.wd_pw, indices=correct_map),
                                correct_map, axis=1)
            img_pw_wt_correct = tf.gather(
                                tf.gather(self.img_pw, indices=correct_map),
                                correct_map, axis=1)
            img_pw_wt_incorrect = tf.gather(
                                tf.gather(self.img_pw, indices=incorrect_map),
                                incorrect_map, axis=1)

            score_correct = utils.weighted_alignment_correlation_tf(
                wd_pw_wt,
                img_pw_wt_correct, 
                pm
                )
            score_incorrect = utils.weighted_alignment_correlation_tf(
                wd_pw_wt,
                img_pw_wt_incorrect, 
                pm
                )

            tot_loss -= (score_correct - score_incorrect)/len(pair_df)
        
        return tot_loss


    def get_month_stats(self):
        """Load reference mean/std alignment loss values sampled by month."""

        sample_means = pd.read_csv(
                    os.path.join(
                        parentdir,
                        "assets/sample_pair_alignments/sample_summaries.csv"
                    ),
                    header=0)

        month_mean = sample_means.loc[
                            sample_means["month"] == int(self.month), "mean"
                            ].iloc[0]
        month_std = sample_means.loc[
                            sample_means["month"] == int(self.month), "std"
                            ].iloc[0]

        return month_mean, month_std


    def loss(self, all_vars, cand_vals, acquired_concepts, remaining_concepts,
             mode="train"):
        """Calculate loss."""

        if mode == "train":
            # Generate temporary partition to backprop loss on                        
            ___, train_idxs, val_idxs, ___ = self.get_partitions(
                            n_concepts=len(self.vocab),
                            acquired_concepts=acquired_concepts,
                            n_required=0, aoa_idx=None,
                            split={"candidate":300,
                                   "train":59,
                                   "val":59,
                                   "test":0})

            self.train_pair_df = self.get_probe_pair_set(
                                                train_idxs, n_pairs=750
                                                )
            self.val_pair_df = self.get_probe_pair_set(
                                                val_idxs, n_pairs=750
                                                )

        if self.sigmoid:
            sig_vars = {k:(tf.sigmoid(v) if k!= "linear_comb" else v)
                         for k, v in all_vars.items()
                        }
        else:
            sig_vars = all_vars
        probs = self.get_probs(sig_vars, cand_vals)
        pm = self.prob_matrix(probs, acquired_concepts, remaining_concepts)

        concept_order = list(
                            acquired_concepts.numpy().astype("int")
                            ) + remaining_concepts

        if mode == "train":
            df = self.train_pair_df
        elif mode == "val":
            df = self.val_pair_df

        loss = self._get_alignment_loss(concept_order, df, pm)

        month_mean, month_std = self.get_month_stats()

        loss = (loss - month_mean)/month_std
        min_loss = (tf.constant(-2, dtype="float32") - month_mean)/month_std # -2 = theoretical minimum value for alignment loss 
        loss = loss - min_loss

        return loss


class SequenceGenerator(FeatureModel):
    """Sequence generator using trained parameter values."""
    def __init__(self, months=None,
                    n_restarts=5, mod_type="optimal", restart=None,
                    var_list=["degree_full"],
                    save_fp =os.path.join(
                            parentdir,
                            "results/run_2/sample_sequences/{}_sequences.csv"), 
                    exclude=[False, True], sigmoid=True, dist_type="eu"):
        
        super().__init__(months, restart, var_list, save_fp, sigmoid,
                         dist_type)

        self.mod_type = mod_type
        self.n_restarts = n_restarts
        self.exclude = exclude
    
        
    def get_best_restart(self):
        """Find best restart based on validation loss."""

        restarts = pd.read_csv(
            os.path.join(
                    parentdir,
                    "results/run_2/model_training/{}_results_{}.csv".format(
                        self.mod_type, 0
                        )),
            header=0, index_col=0 
            )

        if self.n_restarts > 1:
            for i in range(1, self.n_restarts):
                df_a = pd.read_csv(
                    os.path.join(
                        parentdir,
                        "results/run_2/model_training/{}_results_{}.csv".format(
                            self.mod_type, i)
                        ),
                    header=0, index_col=0)
                
                restarts = pd.concat([restarts, df_a])

        # Find restart with min validation loss across last 5 epochs
        min_loss = restarts.loc[
                restarts['epoch'] >= restarts['epoch'].max() - 5 
            ][["restart", "val_loss"]].groupby(
                "restart"
            ).agg(
                "mean"
            ).reset_index()

        re = min_loss.loc[
            min_loss["val_loss"] == min_loss["val_loss"].min(), "restart"
            ].iloc[0]
        self.re = re
        res = restarts.loc[restarts["restart"] == re]
        self.res = res
 
    def process_weights(self):
        """Process weights column into multiple columns."""

        # Process linear weights
        # Replace double spaces
        while any(["  " in x for x in self.res["linear_comb"]]):
            self.res["linear_comb"] = self.res["linear_comb"].str.replace(
                '  ', ' ', regex=True
                )

        # Add commas
        self.res["linear_comb"] = self.res["linear_comb"].str.replace(
                                                        '\[ ', '[', regex=True
                                                        )
        self.res["linear_comb"] = self.res["linear_comb"].str.replace(
                                                        ' \]', ']', regex=True
                                                        )
        self.res["linear_comb"] = self.res["linear_comb"].str.replace(
                                                          ' ', ',', regex=True
                                                        )
        self.res["linear_comb"] = [x[2:-2] for x in self.res["linear_comb"]]
        self.res["linear_comb"] = self.res["linear_comb"].str.replace(
                                                          '\[', '', regex=True
                                                        )
        self.res["linear_comb"] = self.res["linear_comb"].str.replace(
                                                         ' ]', ']', regex=True
                                                        )

        # Split weight list into weight columns
        self.res[
            ["w_" + x for x in self.var_list]
            ] = self.res["linear_comb"].str.split(",", expand=True)

    def load_target_vars(self):
        """Load target variables from best restart."""
        all_vars={}
        lc = []
        
        # Get candidate var values for vars of interest
        for v in self.var_list:

            all_vars[v] = tf.constant(self.res[v].iloc[-1],
                                      name=v, dtype="float32")

            lc.append(float(self.res["w_" + v].iloc[-1]))

        all_vars["linear_comb"] = tf.constant(
                                    [lc], dtype="float32", name="linear_comb"
                                    )
        self.all_vars = all_vars

    def generate_seqs(self, n_seqs):
        """Generate sequences, adding one concept at a time."""

        self.get_best_restart()
        self.process_weights()
        self.load_target_vars()

        cond_dict = {
            True: "ExcAoA", False:"IncAoA"
            }

        mod_dict = {
            "optimal": "synthOpt{}", # Optimal
            "probMatch": "synthProb{}" # AoA (optimal)
        }


        headers = pd.DataFrame(columns=[
            "pid", "condition", "month", "concept", "concept_id",
            "aoa", "acq_idx", "restart"
            ])

        self.cis_stash = [x for x in self.candidate_idxs]

        for exclude in self.exclude:
            start=0

            if not os.path.exists(self.save_fp):
                headers.to_csv(self.save_fp.format(
                    (mod_dict[self.mod_type].format(cond_dict[exclude]))
                ))
            else:
                done = pd.read_csv(self.save_fp.format(
                    (mod_dict[self.mod_type].format(cond_dict[exclude]))
                ), header=0, index_col=0)
                start = done["pid"].max()

            if exclude is True:
                self.candidate_idxs = [x
                                       for x in self.cis_stash
                                       if self.vocab[x] not in list(
                                           self.aoa_dat["concept_i"]
                                           )
                                        ]

            else:
                self.candidate_idxs = [x for x in self.cis_stash]
            
            for s in range(start, n_seqs):
                print("Seq: {}".format(s))
                acquired_concepts = None
                months = []

                for month in self.months:
                    print("Month: {}".format(month))

                    self.month = month
                    n_acq = self.mean_concepts_acquired[month]

                    for i in range(n_acq):
                        print("i: {}".format(i))

                        cand_vals = {}

                        # Get candidate var values for vars of interest
                        for v in self.var_list:
                            function_name = "_get_candidate_" + v
                            args = [acquired_concepts]

                            cand_vals[v] = getattr(self, function_name)(*args)

                        remaining_concepts = self.get_remaining_concepts(
                                                            acquired_concepts
                                                            )
                        acquired_concepts = self.get_next_item(
                                                            self.all_vars,
                                                            cand_vals,
                                                            acquired_concepts,
                                                            remaining_concepts
                                                            )

                        months.append(month)

                acquired = list(np.squeeze(acquired_concepts.numpy()))
                
                acquired_wd = [self.vocab[int(x)] for x in acquired]
                aoa = [x in list(self.aoa_dat["concept_i"])
                       for x in acquired_wd]
                seq_ls = [s] * len(acquired)
                order = [i for i in range(len(acquired))]

                df = pd.DataFrame({
                    "pid": seq_ls,
                    "condition": [
                        mod_dict[self.mod_type].format(cond_dict[exclude])
                        ] * len(acquired),
                    "month": months,
                    "concept": acquired_wd,
                    "concept_id": acquired,
                    "aoa": aoa,
                    "acq_idx": order,
                    "restart": [self.re] * len(acquired)
                })

                df.to_csv(self.save_fp.format(
                    (mod_dict[self.mod_type].format(cond_dict[exclude]))
                    ),
                        mode="a", header=None) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = vars(parser.parse_args())
    print(args)

    """
    mode = args.mode # "train" or "seq_gen"
    model = args.model # "optimal" or "probMatch"
    if args.restart is not None:
        restart = int(args.restart)
    """

    mode="seq_gen"
    model="optimal"
    restart=4

    """python3 /home/kaarina/projects/age_of_acquisition/AoA_final/python/models.py --mode train --model optimal --restart 0"""


    var_list = [
        "degree_subs",
        "degree_full",
        "mean_dist_subs",
        #"mean_dist_full",
        "min_dist_subs",
        "min_dist_full",
        "degree_skew_subs",
        "degree_skew_full",
        "coverage"
    ]

    if mode == "train":
        if model == "optimal":
            opt_trainer = Optimal(
                                var_list=var_list,
                                restart=restart,
                                months=[str(x) for x in range(16, 25)])
            opt_trainer.train_cc(n_epochs=100)

        elif model == "probMatch":
            prob_trainer = ProbMatch(
                                var_list=var_list,
                                restart=restart,
                                months=[str(x) for x in range(16, 25)])
            prob_trainer.train_cc(n_epochs=100)

    if mode == "seq_gen":
        seq_generator = SequenceGenerator(n_restarts=5, mod_type=model,
                                        var_list=var_list,
                                        exclude=[True, False],
                                        months=[str(x) for x in range(16, 25)]
                                        )
    
        seq_generator.generate_seqs(n_seqs=100)

