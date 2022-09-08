import os, sys, inspect
parentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe()))
    )
currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))


sys.path.insert(0,parentdir)

import numpy as np
import pandas as pd

def clean_linear_comb(col):
    """Strip weights column in training df."""
    while any(["  " in x for x in col]):
        col = col.str.replace('  ', ' ', regex=True)
    col = col.str.replace('\[ ', '[', regex=True)
    col = col.str.replace(' \]', ']', regex=True)
    col = col.str.replace(' ', ',', regex=True)
    col = pd.Series([x[2:-2] for x in col])
    col = col.str.replace('\[', '', regex=True)
    col = col.str.replace(' ]', ']', regex=True)
    return col


def get_response_entropy(df_in, entropy="by_month"):
    """Calculate entropy of responses by block or overall."""
    df = df_in.copy()
    c_month = df[
                    ["pid", "condition", "month", "concept"]
                ].groupby(
                    ["pid", "condition", "month"]
                ).agg("count")
    c_month = c_month.reset_index().rename(columns={
        "concept":"n_concepts"
    })

    # Get blank values for each house in each block for each pid
    # This is to have 0 next to houses that were not clicked on each
    cat_list = list(set(df["category"]))

    if entropy == "by_month":
        all_cats = pd.concat(
                            [df[
                                ["pid", "condition", "month"]
                            ].drop_duplicates()]*len(cat_list))
        n_reps = int(len(set(zip(df['pid'],df['condition'], df["month"]))))
        all_cats["category"] = list(np.repeat(cat_list, n_reps))
        all_cats["concept"] = 0 # Chosen 0 times, as default

        # Count n times each house is chosen per block
        df = df[
                ["pid", "condition", "concept", "category", "month"]
             ].groupby(
                ["pid", "condition", "category", "month"]
                ).agg("count")
        df.reset_index(inplace=True)
        df = pd.concat([df, all_cats])
        df = df.groupby(["pid", "condition", "month", "category"]).agg("sum")
        df.reset_index(inplace=True)

        df = df.merge(c_month, on=["pid", "condition", "month"])
        df["cat_p"] = df["concept"]/df["n_concepts"]    # Number of concepts per month
        df["log_p"] = [0 if x == 0 else np.log(x) for x in df["cat_p"]]
        df = df.groupby(
                ["pid", "condition", "month"]
            ).apply(
                lambda x: -sum(x["cat_p"] * x["log_p"])
                )
        df = df.reset_index()
        df.columns = ["pid", "condition", "month", "entropy"]
        df = df[
                    ["pid", "condition", "month", "entropy"]
                ].groupby(
                    ["pid", "condition", "month"]
                ).agg("mean").reset_index()

    if entropy == "to_date":

        ent = pd.DataFrame(columns=["pid", "condition", "month", "entropy"])
        for pid, condition, month in set(
                zip(df['pid'],df['condition'], df["month"])
            ):

            interim = df.loc[
                            (df["pid"] == pid)
                            & (df["condition"] == condition)
                            & (df["month"] <= month)]

            all_cats = pd.DataFrame({"category":cat_list,
                                     "concept":[0]*len(cat_list)})
            interim = interim[
                ["category", "concept"]
                ].groupby("category").agg("count").reset_index()
            interim = pd.concat([interim, all_cats])
            interim = interim.groupby(["category"]).agg("sum")

            interim["cat_p"] = interim["concept"]/interim["concept"].sum()
            interim["log_p"] = [
                0 if x == 0 else np.log(x) for x in interim["cat_p"]
                ]
            entropy = -np.sum(
                [y * x for x, y in zip(interim["log_p"], interim["cat_p"])]
                )

            ent = ent.append(pd.DataFrame({
                "pid": [pid],
                "condition":[condition],
                "month": [month],
                "entropy":[entropy]
            }))
        df = ent

    return df



def get_bootstrapped_pcs(df, month, vocab):
    """Transform bootstrapped distributions into format for y_true."""
    # Get validation set
    y = df.loc[df["month"] == month, ["bootstrap", "concept", "pc"]]
    y = y.pivot_table(index="bootstrap", columns=["concept"])
    y = y.droplevel(0,1)
    y[[x for x in vocab if x not in y.columns]] = 0
    # Reorder columns to align with vocab filt
    y = y[vocab]

    y = np.array(y)
    return y


def calculate_logprob(model, x, y):
    """Get  loglikelilihoods from logprob from regression model."""

    lps = model.predict_log_proba(x)
    logliks = np.take_along_axis(lps, y.astype(int), axis=1)
    return logliks

def calculate_aic(n, loglik):
    aic = 2*n - 2*np.sum(loglik)
    return aic

def calculate_bic(n, d, loglik):
    bic = d*np.log(n) - 2*np.sum(loglik)
    return bic

def get_nonzero_str(x):
    varnames = [v for v in x.index if v not in ["C", "acc", "precision", "f1", "recall"]]
    st = ""
    for v in varnames:
        if x[v] != 0:
            st = st + v
    return st

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