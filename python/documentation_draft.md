# GENERATE_SAMPLE_SEQUENCES

## SeqGenerator class
Class to generate simulated concept learning trajectories, according to the specified sampling procedure.

### Arguments
 - **full_vocab:** (list of str) The full vocabulary from which sequences are sampled. In the case of an alignment project, this would be the list of words included in the intersection of the relevant embeddings. 
 - **sequence_types:** (list of str) The types of sequence being generated using this generator. Can be any subset of ["AoA", "controlIncAoA", "controlExcAoA"]. AoA samples based on children's acquisitions probabilities from WordBank (Frank et al, 2017). controlIncAoA samples randomly from the set of concepts which includes early-acquired concepts (those occurring in WordBank data). controlExcAoA samples randomly from the set of concepts which excluded items occurring in WordBank. Defaults to ["AoA"]

### ***generate*** method
 - **save_fp:** filepath to save results
 - **n_sequences:** Number of sequences to generate. Defaults to 100


### ***_get_monthwise_concepts*** method
Get avg num concepts known/acquired by month from AoA data. 


### ***_get_item_probs*** method
Load set of items and respective probabilities for sampling. Used within generate method to load the appropriate probabilities for the next month of sequence generation.
 - **condition:** Str, one of "AoA", "controlIncAoA" or "controlExcAoA"
 - **acquired_concepts** List of int, indexes of items concepts which have already been acquired in self.vocab


# PROBE_PAIR_EXPT

## ProbePairExpt class
Class to run probe pair experiment from files containing generated sequences.

### Arguments
 - **emb1, emb2:** (arrays) Arrays with dimensions NxD (N=concepts in intersection, D=dimensions of embedding)
 - **vocab1, vocab2:** (lists) Lists of length N. Vocabulary for concepts in each embedding set. the order of items in emb1 corresponds with the order of items in vocab1, and the same if true for vocab2 and emb2. These do not need to be aligned with eachother, nor do the vocabularies need to contain the same set of items.

### ***run*** method
Run the probe pair experiment, and save results to file. 
 - **knowledge_template_fp:** (Str) Formattable string, filepath template from which we can read in the appropriate sequence file when the knowledge condition is inserted. 
 - **results_fp:** (Str) Path to which we save results of probe pair experiment.
 - **knowledge_conditions:** (list) List of knowledge state conditions for which to run the probe pair experiment. Each string in the list is inserted into the knowledge_template_fp (via .format()) to read in the sequences used for the experiment. 
 - **n_probe_reps:** (int) Number of probe pairs to test in each probe pair condition. Defaults to 100.
 - **months**: (None or List, default None). If None, all months available in the sequence file are loaded and tested with probe pairs. If not None, may be a list containing a subset of the months which exist in sequence data.


# GENERATE_FEATURES

## FeatureGenerator class
Class to generate structural features for knowledge states generated as part of simulated concept acquisition trajectories.

### Arguments 
 - **emb1, emb2:** (arrays) Arrays with dimensions NxD (N=concepts in intersection, D=dimensions of embedding)
 - **vocab:** (list) The shared vocabulary for the two input embeddings. Item positions in vocab are aligned with entries in emb1 and emb2. For example, if vocab[0] == "apple", then emb1[0,:] is the embedding for "apple" in system 1, and emb2[0,:] is the embedding for "apple" in system 2.
 - **sequences:** (DataFrame) Dataframe of sequences, for examples as generated from generate_sample_sequences.py's SeqGenerator class. Sequences will be grouped by the "condition" column, so multiple output dfs from SeqGenerator can be concatenated and inputted as one. 
  - **q:** The quantile for connections to be retained in graph construction

### ***_get_stripped_varnames*** method:
Strip column names of modality-specific information, and return list of 'core' column names (i.e, metric which have been calculated for each modality.)
 - **df:** (dataframe) dataframe for which stripped column names are returned


### ***_get_graphs*** method:
Get graphs which retain proportion=q connections for both inputted embeddings.
  - **q:** The quantile for connections to be retained in graph construction


### ***_get_modality_averages*** method:
Return dataframe which included columns that have averages for each metric across modalities.
Input:
 - **df:** (dataframe) dataframe to which cross-modal columns are to be added


### ***_process_pairwise_matrices*** method:
Generate a dataframe of basic structural measures from the pairwise distance matrices of each embedding. Metrics calculated here are:
 - Knowledge-base average of the mean, maximum and minimum distances to another concept in the full concept space
 - Knowledge-base average of the mean, maximum and minimum distances to another concept in the knowledge base
 - The skew of the distribution of pairwise distances for knowledge-base items to fullspace items
 - The skew of the distribution of pairwise distances for knowledge-base items to other knowledge-base items
 - The mean dimensional coverage of the knowledge base (i.e, the mean proportion of a dimension's full range which is covered by the range of concepts in the knowledge base on that dimension)

### ***_generate_basic_graph_measures*** method:
Generate a dataframe of basic structural measures from the graph each embedding. Metrics calculated here are:
- Average min, max and mean degree for knowledge-base items in the full space
- Average min, max and mean degree for knowledge-base items in the knowledge-base


### ***generate_non_nx_measures*** method:
Calls both _process_pairwise_matrices() and _generate_basic_graph_measures, and compiles results (in the instance that generate_nx_measure is too computationally demanding, or should be split out.)

### ***generate_nx_measures*** method:
Generates network measures using networkx for graphs.
 - Clustering (for each modality, and average across modalities)
 - Betweenness (for each modality, and average across modalities)








