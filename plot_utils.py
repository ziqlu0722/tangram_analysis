import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import tangram as tg
import scanpy as sc
import ast
from matplotlib.cm import get_cmap
import matplotlib.ticker as ticker


METRIC_LIST = ["avg_test_score", "sp_sparsity_score", "auc_score"]


def cell_ann_tbl(data, ax, column_headers=None, row_headers=None, scale=(1, 1.5)):
    column_headers = column_headers
    row_headers = row_headers

    rcolors = plt.cm.BuPu(np.full(len(row_headers), 0.1))
    if column_headers is None:
        ccolors = None
    else:
        ccolors = plt.cm.BuPu(np.full(len(column_headers), 0.1))

    table = ax.table(
        cellText=data,
        rowLabels=row_headers,
        rowColours=rcolors,
        rowLoc="right",
        cellLoc="center",
        colColours=ccolors,
        colLabels=column_headers,
        loc="center",
    )
    table.scale(*scale)
    table.set_fontsize(12)
    ax.axis("off")


def pol_coords(xs, ys, pol_deg=2):
    # Fit polynomial'
    pol_cs = np.polyfit(xs, ys, pol_deg)  # polynomial coefficients
    pol_xs = np.linspace(0, 1, 10)  # x linearly spaced
    pol = np.poly1d(pol_cs)  # build polynomial as function
    pol_ys = [pol(x) for x in pol_xs]  # compute polys

    # if real root when y = 0, add point (x, 0):
    roots = pol.r
    root = None
    for i in range(len(roots)):
        if np.isreal(roots[i]) and roots[i] <= 1 and roots[i] >= 0:
            root = roots[i]
            break

    if root is not None:
        pol_xs = np.append(pol_xs, root)
        pol_ys = np.append(pol_ys, 0)

    np.append(pol_xs, 1)
    np.append(pol_ys, pol(1))

    # remove point that are out of [0,1]
    del_idx = []
    for i in range(len(pol_xs)):
        if pol_xs[i] < 0 or pol_ys[i] < 0 or pol_xs[i] > 1 or pol_ys[i] > 1:
            del_idx.append(i)

    pol_xs = [x for x in pol_xs if list(pol_xs).index(x) not in del_idx]
    pol_ys = [y for y in pol_ys if list(pol_ys).index(y) not in del_idx]

    return pol_xs, pol_ys


def auc_scatter_plot(xs, ys, ax, c, scatter=False):
    pol_xs, pol_ys = pol_coords(xs, ys)

    ax.plot(pol_xs, pol_ys, c="r")
    if scatter is True:
        ax.scatter(xs, ys, alpha=0.3, c=c, edgecolors="face")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_aspect(1)
    ax.set_xlabel("score")
    ax.set_ylabel("spatial sparsity")
    ax.tick_params(axis="both", labelsize=8)


def project_cell_ann(adata_map, annotation, threshold=0.5):
    df = tg.one_hot_encoding(adata_map.obs[annotation])
    if "F_out" in adata_map.obs.keys():
        df_ct_prob = adata_map[adata_map.obs["F_out"] > threshold]

    df_ct_prob = adata_map.X.T @ df
    df_ct_prob.index = adata_map.var.index

    return df_ct_prob


def cell_ann_plot(
    adata_map,
    annotation,
    ann=None,
    nrows=1,
    ncols=1,
    x="x",
    y="y",
    cmap="viridis",
    robust=False,
    perc=0,
    invert_y=True,
    s=5,
):

    if not robust and perc != 0:
        raise ValueError("Arg perc is zero when robust is False.")

    if robust and perc == 0:
        raise ValueError("Arg perc cannot be zero when robust is True.")

    df_annotation = project_cell_ann(adata_map, annotation=annotation)
    if ann is not None:
        df_annotation = df_annotation[[ann]]

    fig, axs = plt.subplots(
        nrows, ncols, figsize=(ncols * 3, nrows * 3), sharex=True, sharey=True
    )

    if ann is None:
        axs_f = axs.flatten()
    elif ann is not None:
        axs_f = [axs]

    if invert_y == True:
        axs_f[0].invert_yaxis()

    if len(df_annotation.columns) > nrows * ncols:
        logging.warning(
            "Number of panels smaller than annotations. Increase `nrows`/`ncols`."
        )

    iterator = zip(df_annotation.columns, axs_f)
    for (ann, ax) in iterator:
        single_cell_ann_plot(
            adata_map, annotation, ann, ax, x, y, cmap, robust, perc, s
        )


def single_cell_ann_plot(
    adata_map,
    annotation,
    ann,
    ax,
    x="x",
    y="y",
    cmap="viridis",
    robust=False,
    perc=0,
    s=5,
):
    df_annotation = project_cell_ann(adata_map, annotation=annotation)
    if ann is not None:
        df_annotation = df_annotation[[ann]]

    xs, ys, preds = tg.ordered_predictions(
        adata_map.var[x], adata_map.var[y], df_annotation[ann]
    )
    if robust:
        vmin, vmax = tg.q_value(preds, perc=perc)
    else:
        vmin, vmax = tg.q_value(preds, perc=0)

    ax.scatter(x=xs, y=ys, c=preds, s=s, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(ann)
    ax.set_aspect(1)
    ax.axis("off")


def load_df(fld, exp_idx, label=None):
    df = pd.read_csv(
        os.path.join(fld, "df_all_genes_{}.csv".format(exp_idx)), index_col=0
    )
    if label is None:
        genes = df[df["label"] != "train"].index.values
    if label is not None:
        df = df[(df["label"] == label) | (df["label"] == "train")]
        genes = df[df["label"] == label].index.values
    return df


def load_df_dict(fld, idx_list, label=None):
    df_dict = {}
    for exp_idx in idx_list:
        df = load_df(fld, exp_idx, label)
        df_dict[exp_idx] = df

    return df_dict


def load_map_dict(fld, idx_list):
    map_dict = {}
    for exp_idx in idx_list:
        ad_map = sc.read_h5ad(os.path.join(fld, "ad_map_{}.h5ad".format(exp_idx)))
        map_dict[exp_idx] = ad_map

    return map_dict


def plot_scatter_auc_tbl(df_dict, size=(14, 8), scale=(1, 2)):
    k1 = list(df_dict.keys())[0]
    num_genes = len(df_dict[k1][df_dict[k1]["is_training"] == False].index.values)
    labels = list(set(df_dict[k1][df_dict[k1]["is_training"] == False]["label"].values))
    if len(labels) == 1:
        if labels[0] == "test":
            c = "C0"
        elif labels[0] == "valid":
            c = "C2"
    else:
        c = "grey"

    rows = 2
    columns = len(df_dict)

    fig = plt.figure(figsize=size)
    grid = plt.GridSpec(rows, columns, wspace=0.25, hspace=0.25)

    data_tbl = []
    colheaders_tbl = []

    for (idx, (exp_idx, df)) in enumerate(df_dict.items()):
        metric_dict, ((pol_xs, pol_ys), (xs, ys)) = tg.eval_metric(df)

        # scatter plot
        ax = fig.add_subplot(grid[0, idx])
        sns.scatterplot(
            data=df,
            x="score",
            y="sparsity_sp",
            hue="label",
            alpha=0.5,
            palette={"train": "C1", "test": "C0", "valid": "C2"},
            s=20,
            ax=ax,
        )
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.legend(loc="lower left")
        # auc curve for test/val
        auc_scatter_plot(xs, ys, ax=ax, scatter=False, c=c)
        ax.set_title("experiment_{}\n# evaluated genes: {}".format(exp_idx, num_genes))

        # tbl data
        data_tbl.append(
            [
                float(metric_dict[k])
                for k in ["avg_test_score", "sp_sparsity_score", "auc_score"]
            ]
        )
        colheaders_tbl.append("exp {}".format(exp_idx))

    # tbl
    ax_tbl = fig.add_subplot(grid[1, :])
    data_tbl = np.array(np.round(data_tbl, 3)).T
    rowheaders_tbl = ["avg score", "sp sparsity score", "auc score"]
    cell_ann_tbl(data_tbl, ax_tbl, colheaders_tbl, rowheaders_tbl, scale)


def paired_metric_hist_plot(df_test, df_val, bins=20):
    fig, axs = plt.subplots(2, 3, figsize=(12, 6), sharex="col", sharey="row")

    metrics = ["avg_test_score", "sp_sparsity_score", "auc_score"]

    for ix, metric in enumerate(metrics):
        axs[0, ix].hist(df_test[metric], bins=bins, color="C0", label="test")
        axs[0, ix].set_title("{}".format(metric))
        axs[1, ix].hist(df_val[metric], bins=bins, color="C2", label="validate")
        axs[1, ix].set_title("{}".format(metric))

    plt.suptitle("Score histogram by metric - test vs. val", fontsize=14)


def scatter_all_metrics(df, x="auc_score", y="sp_sparsity_score", c="avg_test_score"):

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    axp = ax.scatter(x=df[x], y=df[y], c=df[c], alpha=0.5, linewidth=0)
    plt.xlabel(x)
    plt.ylabel(y)
    cbar = plt.colorbar(axp, ax=[ax], location="right")
    cbar.set_label(c, labelpad=3)


def trend_all_metrics(df, by="avg_test_score"):

    df = df.sort_values(by=by).reset_index()
    df[["avg_test_score", "sp_sparsity_score", "auc_score"]].plot.line(
        figsize=(15, 3), subplots=False
    )

    plt.legend(bbox_to_anchor=(1.1, 1), loc="upper left", ncol=1)
    plt.gca().yaxis.set_label_position("right")
    plt.gca().yaxis.set_ticks_position("right")
    plt.box(on=None)


def heatmap_all_metrics(df):
    idx_list = get_idx_list(df)
    plt.figure(figsize=(10, 6))
    df = df[["auc_score", "sp_sparsity_score", "avg_test_score"]]
    df = df.sort_values(by="avg_test_score")

    df_norm = (df - df.min()) / (df.max() - df.min())
    ax = sns.heatmap(df_norm.T, cmap="PiYG")

    ax.xaxis.set_major_locator(ticker.LinearLocator(3))
    ax.set_xticklabels(["exp {}".format(ix) for ix in idx_list])
    plt.yticks(rotation=0)


def get_idx_list(df):
    df = df.sort_values(by="avg_test_score")
    ranked_idx_list = list(df["exp_idx"])
    top_idx = ranked_idx_list[-1]
    bottom_idx = ranked_idx_list[0]
    mid_idx = ranked_idx_list[len(ranked_idx_list) // 2]

    idx_list = [bottom_idx, mid_idx, top_idx]

    return idx_list


def overlay_auc(df_dict, df_markers=None, ann=None, top_n=500, cname="Set1"):
    """
    overlay multiple auc curves
    """
    cmap = get_cmap(cname)
    color = iter(cmap.colors)

    plt.figure(figsize=(6, 6))
    #     plt.gca().set_aspect(1)

    for idx, df in df_dict.items():
        if df_markers is None and ann is None:
            genes = None
            k1 = list(df_dict.keys())[0]
            num_genes = len(
                df_dict[k1][df_dict[k1]["is_training"] == False].index.values
            )
        else:
            genes = gen_evaluated_genes(df_markers, df, top_n, ann)
            num_genes = len(genes)

        metric_dict, ((pol_xs, pol_ys), (xs, ys)) = tg.eval_metric(df, genes)
        score = metric_dict["auc_score"]
        c = next(color)
        plt.plot(
            pol_xs,
            pol_ys,
            c=c,
            label="exp {}: score {:.3f} #genes: {}".format(idx, score, num_genes),
        )

    plt.title("AUC curves - {} genes".format(ann))
    plt.legend(loc="lower left")
    plt.xlabel("score")
    plt.ylabel("spatial sparsity")


def cell_ann_deepdive(
    adata_map,
    df_all_genes,
    train_genes,
    df_markers,
    ann,
    top_n=500,
    column_headers=None,
    annotation="cell_type",
    size=(6, 12),
):

    fig, axs = plt.subplots(3, 1, figsize=size)
    fig.tight_layout(pad=7)
    axs_f = axs.flatten()

    genes = gen_evaluated_genes(df_markers, df_all_genes, top_n, ann)
    metric_dict, ((pol_xs, pol_ys), (xs, ys)) = tg.eval_metric(
        df_all_genes, test_genes=genes
    )

    data_tbl = []
    for k in ["avg_test_score", "sp_sparsity_score", "auc_score"]:
        data_tbl.append([metric_dict[k]])
    data_tbl = np.array(np.round(data_tbl, 3))
    rowheaders_tbl = ["avg score", "sp sparsity score", "auc score"]
    colheaders_tbl = ["Scores ({} cells)".format(len(genes))]
    cell_ann_tbl(
        data_tbl,
        axs_f[0],
        column_headers=colheaders_tbl,
        row_headers=rowheaders_tbl,
        scale=(0.6, 1.7),
    )
    auc_scatter_plot(xs, ys, ax=axs_f[1], c="C0", scatter=True)
    single_cell_ann_plot(adata_map, annotation, ann, ax=axs_f[2])

    axs_f[1].set_title("AUC curve ({} cells)".format(len(genes)))
    axs_f[2].set_title("Cell annotation plot")
    box = axs_f[0].get_position()
    axs_f[0].set_position([box.x0 + 0.09, box.y0 - 0.05, box.width, box.height])

    summary = calc_cell_ann_weight(df_markers, train_genes)
    perc = summary.loc[ann]

    plt.suptitle("{} Deepdive".format(ann))
    plt.figtext(
        0.5,
        0.9,
        "train genes labeled as {}: \n{}%".format(ann, perc),
        fontsize=12,
        color="b",
        horizontalalignment="center",
    )


def plot_gene_score_by_annot(
    df_markers, df_all_genes, top_n=500, row=1, col=1, size=(20, 10)
):

    labels = list(
        set(df_all_genes[df_all_genes["is_training"] == False]["label"].values)
    )
    if len(labels) == 1:
        if labels[0] == "test":
            c = "C0"
        elif labels[0] == "valid":
            c = "C2"
    else:
        c = "grey"

    fig, axs = plt.subplots(
        row, col, figsize=size, sharex=True, sharey=True, constrained_layout=True
    )
    axs_f = axs.flatten()

    for idx, col in enumerate(df_markers.columns.values):

        genes = gen_evaluated_genes(df_markers, df_all_genes, top_n, col)
        metric_dict, ((pol_xs, pol_ys), (xs, ys)) = tg.eval_metric(
            df_all_genes, test_genes=genes
        )

        axs_f[idx].set_xlim([0.0, 1.0])
        axs_f[idx].set_ylim([0.0, 1.0])
        axs_f[idx].set_title("{}\n# evaluated genes: {}".format(col, len(genes)))
        axs_f[idx].set_aspect("equal")

        auc_scatter_plot(xs, ys, ax=axs_f[idx], c=c, scatter=True)

        textstr = "\n".join(
            ["{}={}".format(k, np.round(metric_dict[k], 3)) for k in METRIC_LIST]
        )
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.3)
        # place a text box in upper left in axes coords
        axs_f[idx].text(
            0.03,
            0.2,
            textstr,
            transform=axs_f[idx].transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=props,
        )


def gen_evaluated_genes(df_markers, df_all_genes, top_n, ann):
    cell_type_markers = [m.lower() for m in df_markers[ann][:top_n]]
    genes = list(
        set(cell_type_markers)
        & set(
            df_all_genes[
                (df_all_genes["label"] == "valid") | (df_all_genes["label"] == "test")
            ].index.values
        )
    )

    return genes


def top_cell_ann_marker(df_dict, df_markers, ann, top_n):

    cell_type_markers = [m.lower() for m in df_markers[ann][:top_n]]

    overlap_test_genes = set(cell_type_markers)
    for idx, df in df_dict.items():
        overlap_test_genes = overlap_test_genes & set(
            list(df[df["label"] != "train"].index.values)
        )

    genes = list(set(cell_type_markers) & overlap_test_genes)
    return genes


def get_train_genes(exp_idx, df_metric):
    train_genes = df_metric["train_genes"].loc[df_metric["exp_idx"] == exp_idx].iloc[0]
    train_genes = ast.literal_eval(train_genes)

    return train_genes


def calc_cell_ann_weight(df_markers, train_genes):

    df_long = pd.melt(df_markers, var_name="cell_type", value_name="gene_name")

    df_long["label"] = [i for i in range(df_markers.shape[0], 0, -1)] * len(
        df_markers.columns
    )

    df_long.columns = df_long.columns.get_level_values(0)
    df_long["gene_name"] = [x.lower() for x in df_long["gene_name"]]
    df_wide = df_long.pivot(index="gene_name", columns="cell_type", values="label")

    summary = df_wide.loc[train_genes].sum()
    summary = np.round(summary / summary.sum() * 100, 3)

    #     return summary.loc[ann]
    return summary


def return_overlap(a, b):
    overlap = list(set(a) & set(b))
    return len(overlap)


def binned_df(df, metric, bins):

    # get binned_df
    binned_df = pd.DataFrame(df[["exp_idx", "train_genes", metric]])
    binned_df["bins"] = list(map(lambda x: str(x), pd.cut(df[metric], bins=bins)))

    # plot histgram
    plt.hist(binned_df[metric], bins=bins)

    return binned_df


def heatmap_train_gene_overlap(bdf, score_bin, annot=True):
    df = bdf[bdf["bins"] == score_bin]
    idx = pd.MultiIndex.from_product([df["exp_idx"].values, df["exp_idx"].values])

    df_out = pd.Series(
        map(
            lambda x, y: return_overlap(
                get_train_genes(exp_idx=x, df_metric=bdf),
                get_train_genes(exp_idx=y, df_metric=bdf),
            ),
            [e[0] for e in idx],
            [e[1] for e in idx],
        ),
        idx,
    ).unstack()

    sns.heatmap(df_out, cmap="YlGnBu", annot=annot)
    plt.title("metric: {}\nbin: {}".format(bdf.columns[1], score_bin), fontsize=14)

    return df_out

