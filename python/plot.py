import os,sys,inspect
sys.path.append(
    '/Users/apple/projects/Age_of_acquisition/AoA_clean/python'
    )
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

"""Plotting functions for results."""

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)


def probeExpt_results_shady(df,
                            conditions=["controlIncAoA", "AoA"],
                            fn="Alignment_experiment.png",
                            probe_conds=["control", "AoA"],
                            months=None):
    """Plot probe pair results by month."""

    df = df.loc[df["given"].isin(conditions)]

    df["given"] = [re.sub("Mode", "", x) for x in df["given"]]
    df["gp"] = df["given"] + df["probe"]
    plottable = df[
                    ["pid", "given", "probe", "month", "correct_choice"]
                    ].groupby(
                        ["pid", "given", "probe", "month"]
                    ).agg("mean").reset_index()

    color_dict = {
        "controlIncAoA": "#ca0020",
        "controlExcAoA": "#ca0020",
        "AoA": "#2683C6",
        "synthIncAoA": "#253848",
        "synthExcAoA": "#253848",
        "synthProbIncAoA": "#253848",
        "synthProbExcAoA": "#253848",
        "synthOptIncAoA": "orange",
        "synthOptExcAoA": "orange",
        "synthProbIncAoAoldvar": "#253848",
        "synthProbExcAoAoldvar": "#253848",
        "synthOptIncAoAoldvar": "orange",
        "synthOptExcAoAoldvar": "orange",

    }

    __, axs = plt.subplots(
                            1,len(probe_conds), facecolor="white",
                            figsize=(12.5*len(probe_conds), 10),
                            sharey=True)

    if months is not None:
        plottable = plottable.loc[plottable["month"].isin(months)]

    for i, probe in enumerate(probe_conds):

        if len(probe_conds) > 1:
            ax = axs[i]
        else:
            ax = axs
        filt = plottable.loc[
            (plottable["probe"] == probe)
            ]
        probe = filt["probe"].iloc[0]
        sns.lineplot(data = filt, x = "month", y="correct_choice",
                     hue="given", palette=color_dict, ax=ax)
        ax.set_title(f"Probe type: {probe}")
        if i == 0:
            ax.set_ylabel("% Correct forced choice")
        ax.set_xlabel("Month")
    plt.legend(title="Knowledge")
    #fig.suptitle(title)
    plt.savefig(fn)


def plot_losses(restarts,
                fn=os.path.join(parentdir,
                                "results/optimal_teacher_results_test.csv"),
                kw="opt", max_ep=None):
    """Plot loss curves for model training."""

    a = pd.DataFrame({})
    for r in restarts:
        df = pd.read_csv(fn.format(r))

        #df = df.loc[df["month_n"] == df["month_n"].max()]
        df = df[
            ["epoch", "loss", "restart"]
            ].groupby(["epoch", "restart"]).agg("mean").reset_index()
        a = pd.concat([a, df], ignore_index=True)

    if max_ep is not None:
        a = a.loc[a["epoch"] <= max_ep]

    plt.figure()
    sns.lineplot(x="epoch", y="loss", hue="restart", data=a)
    plt.savefig(os.path.join(
        parentdir,
        f"results/run_2/model_training/loss_{kw}.png"))
    plt.show()
