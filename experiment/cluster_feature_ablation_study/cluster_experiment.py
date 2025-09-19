import argparse, json, pickle, re, warnings, sys
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, pairwise_distances
import hdbscan
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------- Load File ----------
def load_dictionary_file(path: Path):
    # Try pickle
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        pass
    # Try JSON
    try:
        txt = path.read_text(encoding="utf-8").strip()
        return json.loads(txt)
    except Exception:
        pass
    # Fallback: eval Python-ish with np.* & Interval(...)
    def _normalize_repr(txt: str) -> str:
        txt = re.sub(r"np\.float(64|32)\(", "(", txt)
        txt = re.sub(r"np\.int(64|32)\(", "(", txt)
        txt = txt.replace("np.nan", "float('nan')")
        txt = re.sub(r"Interval\(([^,]+),\s*([^,]+),\s*closed='[^']+'\)", r"(\1, \2)", txt)
        return txt
    try:
        txt = _normalize_repr(path.read_text(encoding="utf-8"))
        obj = eval(txt, {"__builtins__": {}}, {"nan": float("nan")})
        return obj
    except Exception as e:
        raise RuntimeError(f"Could not parse {path}. Last error: {e}")

def extract_rows(obj):
    """
    Return the list of column-profile dicts from many possible shapes:
    - a list of dicts
    - a dict with keys like 'columns','profiles','data','rows'
    - a nested dict where the list is deeper (we'll search recursively)
    """
    # direct list
    if isinstance(obj, list):
        # must look like a list of dicts
        if obj and isinstance(obj[0], dict):
            return obj
        # empty list is acceptable
        if len(obj) == 0:
            return obj

    # direct dict with common keys
    if isinstance(obj, dict):
        for k in ("columns", "profiles", "data", "rows"):
            v = obj.get(k)
            if isinstance(v, list):
                return v

    # ---- Recursive search: find the first list of dicts anywhere ----
    def _find_list_of_dicts(x):
        if isinstance(x, list):
            if not x:
                return x  # empty list
            if isinstance(x[0], dict):
                return x
            # list but not list of dicts -> no
            return None
        if isinstance(x, dict):
            # try common keys first for speed
            for k in ("columns", "profiles", "data", "rows"):
                if k in x and isinstance(x[k], list) and (not x[k] or isinstance(x[k][0], dict)):
                    return x[k]
            # search all values
            for k, v in x.items():
                res = _find_list_of_dicts(v)
                if res is not None:
                    return res
        # other types -> no
        return None

    found = _find_list_of_dicts(obj)
    if found is not None:
        return found

    # ---- Debug aid: show a preview of top-level keys/types ----
    if isinstance(obj, dict):
        preview = {k: type(v).__name__ for k, v in obj.items()}
        raise ValueError(f"Couldn't find list of column profile dicts. Top-level keys: {preview}")
    raise ValueError(f"Couldn't find list of column profile dicts; got type {type(obj).__name__}.")



# ---------- Feature engineering ----------
NUMERIC_COLS_ALL = [
    # existing
    "null_ratio","distinct_num","unique_ratio",
    "words_length_avg","cells_length_avg","cells_null",
    "numeric_min","numeric_max","Q1","Q2","Q3",
    "most_freq_value_ratio","max_digits","max_decimals",
    "max_len","min_len","avg_len",
    # new numeric you introduced
    "row_num",
    # words_info
    "words_unique","words_alphabet","words_numeric","words_punctuation","words_miscellaneous",
    # cell_info
    "cells_unique","cells_alphabet","cells_numeric","cells_punctuation","cells_miscellaneous",
    # character_info
    "characters_unique","characters_alphabet","characters_numeric","characters_punctuation","characters_miscellaneous",
]

CATEG_COLS_ALL = [
    # existing categorical bases
    "basic_data_type","semantic_domain","dominant_pattern","first_digit",
]

ALL_COLS = NUMERIC_COLS_ALL + CATEG_COLS_ALL

FEATURE_GROUPS = {
    "stats_basic": ["row_num","null_ratio","distinct_num","unique_ratio"],
    "distribution": ["histogram","histogram_freq", "equi_width_bin", "equi_depth_bin","numeric_min","numeric_max","Q1","Q2","Q3","most_freq_value_ratio","first_digit",
    "max_decimals"],
    "type_info": ["basic_data_type"],
    "pattern_info": ["dominant_pattern", "max_digits", "max_decimals", "avg_len", "min_len", "max_len"],
    "semantic_domain":["semantic_domain"],
    "keywords": "DYNAMIC",  # vectorize_top_keywords
    "character_info": ["characters_unique", "characters_alphabet","characters_numeric", "characters_punctuation",
                       "characters_miscellaneous"],
    "words_info": ["words_unique", "words_alphabet","words_numeric","words_punctuation","words_miscellaneous","words_length_avg"],
    "cell_info": [ "cells_unique","cells_alphabet","cells_numeric","cells_punctuation","cells_miscellaneous","cells_length_avg"],

}



def _best_k_by_silhouette(X, k_grid, metric="euclidean"):
    """
    Returns (best_k, best_score). Silhouette needs k >= 2 and k < n_samples.
    For cosine, L2-normalize rows before scoring (recommended).
    """
    n = X.shape[0]
    valid_ks = [k for k in k_grid if 2 <= k < n]
    if not valid_ks:
        return None, None

    X_eval = X
    if metric == "cosine":
        # silhouette_score with cosine works better on normalized vectors
        X_eval = normalize(X, norm="l2", axis=1, copy=False)

    best_k, best_s = None, -1.0
    for k in valid_ks:
        try:
            # We don't need labels here; silhouette uses pairwise distances internally
            # But we must produce labels to compute it; create temporary KMeans fit
            km_tmp = KMeans(n_clusters=k, random_state=42, n_init=20).fit(X)
            labels = km_tmp.labels_
            s = silhouette_score(X_eval, labels, metric=metric)
            if s > best_s:
                best_k, best_s = k, s
        except Exception:
            # skip unstable k values
            pass
    return best_k, best_s

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans

def _kmeans_inertia_curve(X, k_grid, random_state=42, n_init=20):
    ks, inertias = [], []
    n = X.shape[0]
    for k in k_grid:
        if 2 <= k < n:
            try:
                km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init).fit(X)
                ks.append(k)
                inertias.append(float(km.inertia_))
            except Exception:
                pass
    return np.array(ks), np.array(inertias)

def _pick_k_by_elbow(ks, inertias):
    """
    Simple knee heuristic: pick k with largest second difference of the
    log inertia curve. Works well enough for automation.
    """
    if len(ks) < 3:
        return ks[0] if len(ks) else None
    y = np.log(np.maximum(inertias, 1e-9))
    # second finite difference
    secdiff = y[:-2] - 2*y[1:-1] + y[2:]
    # index of max curvature point maps to k at i+1
    i = int(np.argmax(secdiff))
    return int(ks[i+1])

def _save_elbow_plot(ks, inertias, out_path, title="KMeans Elbow (inertia vs k)"):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(ks, inertias, marker="o")
    ax.set_xlabel("k")
    ax.set_ylabel("inertia (within-cluster sum of squares)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans

# def plot_elbow_and_silhouette(
#     X, ks, out_prefix: Path, random_state=42, n_init=20, log_inertia=True
# ):
#     """
#     Saves ONE figure:
#       - f"{out_prefix}_elbow_silhouette.png"
#
#     Left y-axis: KMeans inertia (optionally log-scaled)
#     Right y-axis: mean silhouette score (only when all clusters have >= 2 points)
#     """
#     from pathlib import Path
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from sklearn.cluster import KMeans
#     from sklearn.metrics import silhouette_score
#
#     out_prefix = Path(out_prefix)
#     out_prefix.parent.mkdir(parents=True, exist_ok=True)
#
#     ks_inertia, inertias = [], []
#     sil_ks, sil_scores = [], []
#
#     n = len(X)
#     for k in ks:
#         if k < 2 or k > n:
#             continue
#         km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
#         labels = km.fit_predict(X)
#         ks_inertia.append(k)
#         inertias.append(float(km.inertia_))
#
#         # silhouette only when no singleton clusters
#         uniq, counts = np.unique(labels, return_counts=True)
#         if len(uniq) >= 2 and np.all(counts >= 2):
#             try:
#                 sil = silhouette_score(X, labels, metric="euclidean")
#                 sil_ks.append(k)
#                 sil_scores.append(float(sil))
#             except Exception:
#                 pass
#
#     if not ks_inertia:
#         return  # nothing to plot
#
#     # --- optional elbow-k via simple knee on log-inertia ---
#     def _pick_k_by_elbow_local(ks_arr, inertias_arr):
#         if len(ks_arr) < 3:
#             return ks_arr[0]
#         y = np.log(np.maximum(inertias_arr, 1e-9))
#         secdiff = y[:-2] - 2*y[1:-1] + y[2:]
#         i = int(np.argmax(secdiff))
#         return int(ks_arr[i+1])
#
#     elbow_k = _pick_k_by_elbow_local(np.array(ks_inertia), np.array(inertias))
#     best_sil_k = sil_ks[int(np.argmax(sil_scores))] if sil_scores else None
#
#     # --- plot ---
#     fig, ax1 = plt.subplots(figsize=(7, 4.5))
#
#     line1, = ax1.plot(ks_inertia, inertias, marker="o")#, label="Inertia"
#     if log_inertia:
#         ax1.set_yscale("log")
#         ax1.set_ylabel("Inertia (log)")
#     else:
#         ax1.set_ylabel("Inertia")
#     ax1.set_xlabel("k")
#     ax1.grid(True, alpha=0.3)
#
#     ax2 = ax1.twinx()
#     if sil_scores:
#         line2, = ax2.plot(
#             sil_ks,
#             sil_scores,
#             marker="s",
#             linestyle="--",
#             color="green"
#
#         )#label="Silhouette"
#         ax2.set_ylabel("Mean silhouette")
#     ax1.tick_params(axis="both", which="major", labelsize=14)
#     ax2.tick_params(axis="both", which="major", labelsize=14)
#     # vertical guides
#     ax1.axvline(elbow_k, linestyle=":", linewidth=1)
#     ax1.text(elbow_k, ax1.get_ylim()[1], f"  elbow k={elbow_k}", va="top",fontsize=14)
#
#     if best_sil_k is not None:
#         ax1.axvline(best_sil_k, linestyle="--", linewidth=1)
#         ax1.text(best_sil_k, ax1.get_ylim()[0], f"  best sil k={best_sil_k}", va="bottom", fontsize=14)
#
#     # combined legend
#     handles = [line1]
#     if sil_scores:
#         handles.append(line2)
#     #ax1.legend(handles, [h.get_label() for h in handles], loc="best", fontsize = 14)
#
#     #plt.title("Elbow & Silhouette vs k")
#     plt.tight_layout()
#     out_path = out_prefix.with_name(out_prefix.name + "_elbow_silhouette.png")
#     fig.savefig(out_path, dpi=150)
#     plt.close(fig)
# def plot_elbow_and_silhouette(
#     X, ks, out_prefix: Path, random_state=42, n_init=20, log_inertia=True
# ):
#     """
#     Saves:
#       - f"{out_prefix}_elbow_silhouette.png"  (Inertia + mean silhouette vs k)
#       - f"{out_prefix}_silhouette_plot.png"   (Silhouette diagram for best k)
#     """
#     from pathlib import Path
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from sklearn.cluster import KMeans
#     from sklearn.metrics import silhouette_score, silhouette_samples
#     import matplotlib.cm as cm
#
#     out_prefix = Path(out_prefix)
#     out_prefix.parent.mkdir(parents=True, exist_ok=True)
#
#     ks_inertia, inertias = [], []
#     sil_ks, sil_scores = [], []
#
#     n = len(X)
#     for k in ks:
#         if k < 2 or k > n:
#             continue
#         km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
#         labels = km.fit_predict(X)
#         ks_inertia.append(k)
#         inertias.append(float(km.inertia_))
#
#         # silhouette only when no singleton clusters
#         uniq, counts = np.unique(labels, return_counts=True)
#         if len(uniq) >= 2 and np.all(counts >= 2):
#             try:
#                 sil = silhouette_score(X, labels, metric="euclidean")
#                 sil_ks.append(k)
#                 sil_scores.append(float(sil))
#             except Exception:
#                 pass
#
#     if not ks_inertia:
#         return  # nothing to plot
#
#     # --- elbow heuristic on inertia ---
#     def _pick_k_by_elbow_local(ks_arr, inertias_arr):
#         if len(ks_arr) < 3:
#             return ks_arr[0]
#         y = np.log(np.maximum(inertias_arr, 1e-9))
#         secdiff = y[:-2] - 2*y[1:-1] + y[2:]
#         i = int(np.argmax(secdiff))
#         return int(ks_arr[i+1])
#
#     elbow_k = _pick_k_by_elbow_local(np.array(ks_inertia), np.array(inertias))
#     best_sil_k = sil_ks[int(np.argmax(sil_scores))] if sil_scores else None
#
#     # --- plot 1: elbow + mean silhouette ---
#     fig, ax1 = plt.subplots(figsize=(7, 4.5))
#
#     ax1.plot(ks_inertia, inertias, marker="o")
#     if log_inertia:
#         ax1.set_yscale("log")
#         ax1.set_ylabel("Inertia (log)", fontsize=12)
#     else:
#         ax1.set_ylabel("Inertia", fontsize=12)
#     ax1.set_xlabel("k", fontsize=12)
#     ax1.grid(True, alpha=0.3)
#
#     ax2 = ax1.twinx()
#     if sil_scores:
#         ax2.plot(
#             sil_ks,
#             sil_scores,
#             marker="s",
#             linestyle="--",
#             color="green"
#         )
#         ax2.set_ylabel("Mean silhouette", fontsize=12)
#
#     # vertical guides
#     ax1.axvline(elbow_k, linestyle=":", linewidth=1)
#     ax1.text(elbow_k, ax1.get_ylim()[1], f"  elbow k={elbow_k}", va="top")
#
#     if best_sil_k is not None:
#         ax1.axvline(best_sil_k, linestyle="--", linewidth=1)
#         ax1.text(best_sil_k, ax1.get_ylim()[0], f"  best sil k={best_sil_k}", va="bottom")
#
#     ax1.tick_params(axis="both", which="major", labelsize=12)
#     ax2.tick_params(axis="both", which="major", labelsize=12)
#
#     plt.title("Elbow & Mean Silhouette vs k", fontsize=13)
#     plt.tight_layout()
#     out_path = out_prefix.with_name(out_prefix.name + "_elbow_silhouette.png")
#     fig.savefig(out_path, dpi=150)
#     plt.close(fig)
#
#     # --- plot 2: silhouette diagram for best k ---
#     if best_sil_k is not None:
#         km = KMeans(n_clusters=best_sil_k, random_state=random_state, n_init=n_init)
#         labels = km.fit_predict(X)
#         sample_sil = silhouette_samples(X, labels)
#
#         fig, ax = plt.subplots(figsize=(7, 5))
#         y_lower = 10
#         for i in range(best_sil_k):
#             ith_sil = sample_sil[labels == i]
#             ith_sil.sort()
#             size_i = len(ith_sil)
#             y_upper = y_lower + size_i
#
#             color = cm.nipy_spectral(float(i) / best_sil_k)
#             ax.fill_betweenx(
#                 np.arange(y_lower, y_upper),
#                 0, ith_sil,
#                 facecolor=color, edgecolor=color, alpha=0.7
#             )
#             ax.text(-0.05, y_lower + 0.5 * size_i, str(i))
#             y_lower = y_upper + 10
#
#         ax.axvline(np.mean(sample_sil), color="red", linestyle="--")
#         ax.set_title(f"Silhouette plot for k={best_sil_k}", fontsize=13)
#         ax.set_xlabel("Silhouette coefficient values", fontsize=12)
#         ax.set_ylabel("Cluster", fontsize=12)
#         ax.tick_params(axis="both", which="major", labelsize=11)
#         plt.tight_layout()
#
#         out_path = out_prefix.with_name(f"{out_prefix.name}_silhouette_plot_k{best_sil_k}.png")
#
#         fig.savefig(out_path, dpi=150)
#         plt.close(fig)

# def plot_elbow_and_silhouette( # heatmap with best silhouette k
#     X, ks, out_prefix: Path, random_state=42, n_init=20, log_inertia=True,
#     sample_names=None,              # list/Series of names for rows in X
#     label_members=False,            # write names inside the silhouette plot
#     max_labels_per_cluster=50       # safety cap to avoid over-plotting
# ):
#
#     """
#     Saves:
#       - f"{out_prefix}_elbow_silhouette.png"          (Inertia + mean silhouette vs k)
#       - f"{out_prefix}_silhouette_plot_k{K}.png"      (Silhouette diagram for best silhouette k)
#       - f"{out_prefix}_assignments_k{K}.csv"          (Row-level cluster assignments + silhouette)
#       - f"{out_prefix}_cluster_summary_k{K}.csv"      (Per-cluster feature summary)
#     """
#     from pathlib import Path
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from sklearn.cluster import KMeans
#     from sklearn.metrics import silhouette_score, silhouette_samples
#     import matplotlib.cm as cm
#
#     # NEW: import pandas only if needed
#     import pandas as pd
#
#     out_prefix = Path(out_prefix)
#     out_prefix.parent.mkdir(parents=True, exist_ok=True)
#
#     # --- ensure we have a DataFrame view for saving ---
#     if isinstance(X, pd.DataFrame):
#         X_df = X.copy()
#         feature_names = list(X_df.columns)
#     else:
#         X_df = pd.DataFrame(X)  # generic columns if numpy
#         feature_names = [f"feature_{i}" for i in range(X_df.shape[1])]
#         X_df.columns = feature_names
#
#     ks_inertia, inertias = [], []
#     sil_ks, sil_scores = [], []
#
#     n = len(X_df)
#     for k in ks:
#         if k < 2 or k > n:
#             continue
#         km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
#         labels = km.fit_predict(X_df.values)
#         ks_inertia.append(k)
#         inertias.append(float(km.inertia_))
#
#         # silhouette only when no singleton clusters
#         uniq, counts = np.unique(labels, return_counts=True)
#         if len(uniq) >= 2 and np.all(counts >= 2):
#             try:
#                 sil = silhouette_score(X_df.values, labels, metric="euclidean")
#                 sil_ks.append(k)
#                 sil_scores.append(float(sil))
#             except Exception:
#                 pass
#
#     if not ks_inertia:
#         return  # nothing to plot
#
#     # --- elbow heuristic on inertia ---
#     def _pick_k_by_elbow_local(ks_arr, inertias_arr):
#         if len(ks_arr) < 3:
#             return ks_arr[0]
#         y = np.log(np.maximum(inertias_arr, 1e-9))
#         secdiff = y[:-2] - 2*y[1:-1] + y[2:]
#         i = int(np.argmax(secdiff))
#         return int(ks_arr[i+1])
#
#     elbow_k = _pick_k_by_elbow_local(np.array(ks_inertia), np.array(inertias))
#     best_sil_k = sil_ks[int(np.argmax(sil_scores))] if sil_scores else None
#
#     # --- plot 1: elbow + mean silhouette ---
#     fig, ax1 = plt.subplots(figsize=(7, 4.5))
#     ax1.plot(ks_inertia, inertias, marker="o")
#     if log_inertia:
#         ax1.set_yscale("log")
#         ax1.set_ylabel("Inertia (log)", fontsize=12)
#     else:
#         ax1.set_ylabel("Inertia", fontsize=12)
#     ax1.set_xlabel("k", fontsize=12)
#     ax1.grid(True, alpha=0.3)
#
#     ax2 = ax1.twinx()
#     if sil_scores:
#         ax2.plot(sil_ks, sil_scores, marker="s", linestyle="--", color="green")
#         ax2.set_ylabel("Mean silhouette", fontsize=12)
#
#     ax1.axvline(elbow_k, linestyle=":", linewidth=1)
#     ax1.text(elbow_k, ax1.get_ylim()[1], f"  elbow k={elbow_k}", va="top")
#     if best_sil_k is not None:
#         ax1.axvline(best_sil_k, linestyle="--", linewidth=1)
#         ax1.text(best_sil_k, ax1.get_ylim()[0], f"  best sil k={best_sil_k}", va="bottom")
#
#     ax1.tick_params(axis="both", which="major", labelsize=12)
#     ax2.tick_params(axis="both", which="major", labelsize=12)
#     plt.title("Elbow & Mean Silhouette vs k", fontsize=13)
#     plt.tight_layout()
#     fig.savefig(out_prefix.with_name(out_prefix.name + "_elbow_silhouette.png"), dpi=150)
#     plt.close(fig)
#
#     # --- plot 2 + EXPORTS: silhouette diagram & CSVs for best silhouette k ---
#     if best_sil_k is not None:
#         km = KMeans(n_clusters=best_sil_k, random_state=random_state, n_init=n_init)
#         labels = km.fit_predict(X_df.values)
#         sample_sil = silhouette_samples(X_df.values, labels)
#
#         # Silhouette diagram
#         # fig, ax = plt.subplots(figsize=(7, 5))
#         # y_lower = 10
#         # for i in range(best_sil_k):
#         #     ith_sil = sample_sil[labels == i]
#         #     ith_sil.sort()
#         #     size_i = len(ith_sil)
#         #     y_upper = y_lower + size_i
#         #
#         #     color = cm.nipy_spectral(float(i) / best_sil_k)
#         #     ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_sil,
#         #                      facecolor=color, edgecolor=color, alpha=0.7)
#         #     # label cluster band with id and size
#         #     ax.text(-0.05, y_lower + 0.5 * size_i, f"{i} (n={size_i})")
#         #     y_lower = y_upper + 10
#         #
#         # ax.axvline(np.mean(sample_sil), color="red", linestyle="--")
#         # ax.set_title(f"Silhouette plot for k={best_sil_k}", fontsize=13)
#         # ax.set_xlabel("Silhouette coefficient values", fontsize=12)
#         # ax.set_ylabel("Cluster", fontsize=12)
#         # ax.tick_params(axis="both", which="major", labelsize=11)
#         # plt.tight_layout()
#         # fig.savefig(out_prefix.with_name(f"{out_prefix.name}_silhouette_plot_k{best_sil_k}.png"), dpi=150)
#         # plt.close(fig)
#         # --- plot 2 + EXPORTS: silhouette diagram & CSVs for best silhouette k ---
#         if best_sil_k is not None:
#             km = KMeans(n_clusters=best_sil_k, random_state=random_state, n_init=n_init)
#             labels = km.fit_predict(X_df.values)
#             sample_sil = silhouette_samples(X_df.values, labels)
#
#             # --- NEW: save cluster -> members (column names) CSV ---
#             if sample_names is None:
#                 # default to index if names not provided
#                 sample_names_arr = np.array([str(i) for i in range(len(X_df))])
#             else:
#                 sample_names_arr = np.asarray(sample_names).astype(str)
#                 if len(sample_names_arr) != len(X_df):
#                     raise ValueError("len(sample_names) must equal number of rows in X")
#
#             members_rows = []
#             for cid in range(best_sil_k):
#                 idx = np.where(labels == cid)[0]
#                 for r in idx:
#                     members_rows.append({
#                         "cluster": cid,
#                         "member": sample_names_arr[r],
#                         "silhouette": float(sample_sil[r])
#                     })
#             members_df = pd.DataFrame(members_rows).sort_values(["cluster", "silhouette"], ascending=[True, False])
#             members_path = out_prefix.with_name(f"{out_prefix.name}_cluster_members_k{best_sil_k}.csv")
#             members_df.to_csv(members_path, index=False)
#
#             # --- Silhouette diagram (with optional member labels) ---
#             fig, ax = plt.subplots(figsize=(8.5, 6))  # a bit wider to fit names
#             y_lower = 10
#             for i in range(best_sil_k):
#                 idx = np.where(labels == i)[0]
#                 ith_sil = sample_sil[idx]
#                 order = np.argsort(ith_sil)  # ascending for the classic ridge look
#                 ith_sil = ith_sil[order]
#                 idx = idx[order]
#
#                 size_i = len(ith_sil)
#                 y_upper = y_lower + size_i
#
#                 color = cm.nipy_spectral(float(i) / best_sil_k)
#                 ax.fill_betweenx(
#                     np.arange(y_lower, y_upper),
#                     0, ith_sil,
#                     facecolor=color, edgecolor=color, alpha=0.7
#                 )
#
#                 # cluster label + size
#                 ax.text(-0.08, y_lower + 0.5 * size_i, f"{i} (n={size_i})", va="center", ha="right", fontsize=10)
#
#                 # --- NEW: draw member names next to each horizontal bar (capped) ---
#                 if label_members:
#                     n_to_label = min(size_i, max_labels_per_cluster)
#                     step = max(1, size_i // n_to_label)
#                     # label every `step`-th sample to reduce clutter
#                     for j in range(0, size_i, step):
#                         y_pos = y_lower + j + 0.5
#                         name = sample_names_arr[idx[j]]
#                         # place name slightly to the right of the bar start (x≈0)
#                         ax.text(0.02, y_pos, name, va="center", fontsize=8)
#
#                 y_lower = y_upper + 10
#
#             ax.axvline(np.mean(sample_sil), color="red", linestyle="--", linewidth=1)
#             ax.set_title(f"Silhouette plot for k={best_sil_k}", fontsize=13)
#             ax.set_xlabel("Silhouette coefficient values", fontsize=12)
#             ax.set_ylabel("Cluster", fontsize=12)
#             ax.tick_params(axis="both", which="major", labelsize=11)
#
#             # give room on the left for names and cluster ids
#             ax.set_xlim(-0.12, 1.0)
#             plt.tight_layout()
#             fig.savefig(out_prefix.with_name(f"{out_prefix.name}_silhouette_plot_k{best_sil_k}.png"), dpi=150)
#             plt.close(fig)
#
#         # --- NEW: Row-level assignments (what's inside each cluster) ---
#         assign = X_df.copy()
#         assign["cluster"] = labels
#         assign["silhouette"] = sample_sil
#         assign_path = out_prefix.with_name(f"{out_prefix.name}_assignments_k{best_sil_k}.csv")
#         assign.to_csv(assign_path, index=True)  # keep index so you can trace back rows
#
#         # --- NEW: Cluster summary (numeric + simple categorical) ---
#         # Separate numeric vs non-numeric
#         num_cols = assign.select_dtypes(include=[np.number]).columns.intersection(feature_names)
#         cat_cols = [c for c in feature_names if c not in num_cols]
#
#         # numeric summary
#         num_summary = assign.groupby("cluster")[list(num_cols)].agg(["count", "mean", "std", "min", "max"])
#
#         # simple categorical summary: mode and its frequency
#         cat_frames = []
#         for c in cat_cols:
#             mode_vals = assign.groupby("cluster")[c].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
#             mode_freq = assign.groupby("cluster")[c].agg(lambda s: s.value_counts().iloc[0] if not s.empty else 0)
#             tmp = pd.DataFrame({"mode": mode_vals, "mode_freq": mode_freq})
#             tmp.columns = pd.MultiIndex.from_product([[c], tmp.columns])
#             cat_frames.append(tmp)
#
#         if cat_frames:
#             cat_summary = pd.concat(cat_frames, axis=1)
#             summary = pd.concat([num_summary, cat_summary], axis=1)
#         else:
#             summary = num_summary
#
#         summary_path = out_prefix.with_name(f"{out_prefix.name}_cluster_summary_k{best_sil_k}.csv")
#         summary.to_csv(summary_path)

from pathlib import Path

def plot_elbow_and_silhouette(
    X, ks, out_prefix: Path, random_state=42, n_init=20, log_inertia=True,
    sample_names=None, label_members=False, max_labels_per_cluster=50,
    k_for_detail: str = "silhouette",   # "silhouette" or "elbow"
):
    """
    Saves:
      - <out_prefix>_elbow_silhouette.png
      - <out_prefix>_silhouette_plot_k{K}_{tag}.png          (if silhouettes valid)
      - <out_prefix>_cluster_members_k{K}_{tag}.csv
      - <out_prefix>_similarity_heatmap_k{K}_{tag}.png
    Where {tag} is "elbow" or "silhouette" based on k_for_detail.
    """
    # stdlib / deps
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, silhouette_samples
    from sklearn.metrics.pairwise import cosine_similarity  # used for heatmap

    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # --- Ensure we have a DataFrame view for consistent handling ---
    if isinstance(X, pd.DataFrame):
        X_df = X.copy()
    else:
        X_df = pd.DataFrame(np.asarray(X))
    n_samples = len(X_df)

    # Optional names
    if sample_names is None:
        names_arr = np.array([str(i) for i in range(n_samples)])
    else:
        names_arr = np.asarray(sample_names).astype(str)
        if len(names_arr) != n_samples:
            raise ValueError("len(sample_names) must equal number of rows in X")

    # --- sweep k: inertia + mean silhouette (when valid) ---
    ks_inertia, inertias = [], []
    sil_ks, sil_scores = [], []

    for k in ks:
        if not (2 <= k <= n_samples):
            continue
        km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = km.fit_predict(X_df.values)
        ks_inertia.append(k)
        inertias.append(float(km.inertia_))

        # mean silhouette only when ≥2 clusters and no singleton clusters
        uniq, counts = np.unique(labels, return_counts=True)
        if len(uniq) >= 2 and np.all(counts >= 2):
            try:
                sil = silhouette_score(X_df.values, labels, metric="euclidean")
                sil_ks.append(k)
                sil_scores.append(float(sil))
            except Exception:
                pass

    if not ks_inertia:
        return  # nothing to plot

    # --- elbow heuristic on (log) inertia via 2nd finite difference ---
    def _pick_k_by_elbow_local(ks_arr, inertias_arr):
        if len(ks_arr) < 3:
            return int(ks_arr[0])
        y = np.log(np.maximum(np.asarray(inertias_arr), 1e-9))
        secdiff = y[:-2] - 2 * y[1:-1] + y[2:]
        i = int(np.argmax(secdiff))
        return int(ks_arr[i + 1])

    elbow_k = _pick_k_by_elbow_local(np.array(ks_inertia), np.array(inertias))
    best_sil_k = int(sil_ks[int(np.argmax(sil_scores))]) if sil_scores else None

    # --- plot 1: elbow + mean silhouette vs k ---
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(ks_inertia, inertias, marker="o")
    if log_inertia:
        ax1.set_yscale("log")
        ax1.set_ylabel("Inertia (log)", fontsize=12)
    else:
        ax1.set_ylabel("Inertia", fontsize=12)
    ax1.set_xlabel("k", fontsize=12)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    if sil_scores:
        ax2.plot(sil_ks, sil_scores, marker="s", linestyle="--", color="green")
        ax2.set_ylabel("Mean silhouette", fontsize=12)

    # vertical guides
    ax1.axvline(elbow_k, linestyle=":", linewidth=1)
    ax1.text(elbow_k, ax1.get_ylim()[1], f"  elbow k={elbow_k}", va="top")
    if best_sil_k is not None:
        ax1.axvline(best_sil_k, linestyle="--", linewidth=1)
        ax1.text(best_sil_k, ax1.get_ylim()[0], f"  best sil k={best_sil_k}", va="bottom")

    ax1.tick_params(axis="both", which="major", labelsize=12)
    ax2.tick_params(axis="both", which="major", labelsize=12)
    plt.title("Elbow & Mean Silhouette vs k", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_prefix.with_name(out_prefix.name + "_elbow_silhouette.png"), dpi=150)
    plt.close(fig)

    # --- choose which k drives the detail artifacts ---
    if k_for_detail == "elbow":
        chosen_k, chosen_tag = elbow_k, "elbow"
    else:
        chosen_k, chosen_tag = best_sil_k, "silhouette"

    if chosen_k is None:
        return  # cannot proceed with details

    # --- fit chosen k and export details ---
    km = KMeans(n_clusters=chosen_k, random_state=random_state, n_init=n_init)
    labels = km.fit_predict(X_df.values)

    # Save cluster members CSV (always)
    rows = [{"cluster": int(labels[i]), "member": names_arr[i]} for i in range(n_samples)]
    pd.DataFrame(rows).sort_values(["cluster", "member"]).to_csv(
        out_prefix.with_name(f"{out_prefix.name}_cluster_members_k{chosen_k}_{chosen_tag}.csv"),
        index=False
    )

    # Try silhouette diagram (requires no singleton clusters)
    uniq, counts = np.unique(labels, return_counts=True)
    can_sil = (len(uniq) >= 2 and np.all(counts >= 2))
    if can_sil:
        sample_sil = silhouette_samples(X_df.values, labels)
        fig, ax = plt.subplots(figsize=(8.5, 6))
        y_lower = 10
        for i in range(chosen_k):
            idx = np.where(labels == i)[0]
            vals = sample_sil[idx]
            order = np.argsort(vals)
            idx, vals = idx[order], vals[order]
            size_i = len(vals)
            y_upper = y_lower + size_i

            color = cm.nipy_spectral(float(i) / chosen_k)
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals,
                             facecolor=color, edgecolor=color, alpha=0.7)
            ax.text(-0.08, y_lower + 0.5 * size_i, f"{i} (n={size_i})",
                    va="center", ha="right", fontsize=10)

            if label_members:
                n_to_label = min(size_i, max_labels_per_cluster)
                step = max(1, size_i // max(1, n_to_label))
                for j in range(0, size_i, step):
                    ax.text(0.02, y_lower + j + 0.5, names_arr[idx[j]],
                            va="center", fontsize=8)

            y_lower = y_upper + 10

        ax.axvline(np.mean(sample_sil), color="red", linestyle="--", linewidth=1)
        ax.set_title(f"Silhouette plot for k={chosen_k} ({chosen_tag})", fontsize=13)
        ax.set_xlabel("Silhouette coefficient values", fontsize=12)
        ax.set_ylabel("Cluster", fontsize=12)
        ax.tick_params(axis="both", which="major", labelsize=11)
        ax.set_xlim(-0.12, 1.0)
        plt.tight_layout()
        fig.savefig(out_prefix.with_name(
            f"{out_prefix.name}_silhouette_plot_k{chosen_k}_{chosen_tag}.png"
        ), dpi=150)
        plt.close(fig)

    # Similarity heatmap ordered by clusters (cosine similarity)
    S = cosine_similarity(X_df.values)
    order = np.argsort(labels)
    S_ord = S[order][:, order]
    names_ord = names_arr[order]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(S_ord, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_title(f"Column similarity heatmap (k={chosen_k}, {chosen_tag})")
    ax.set_xticks(np.arange(len(names_ord)))
    ax.set_yticks(np.arange(len(names_ord)))
    ax.set_xticklabels(names_ord, rotation=90, fontsize=6)
    ax.set_yticklabels(names_ord, fontsize=6)
    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.ax.tick_params(labelsize=8)
    plt.tight_layout()
    fig.savefig(out_prefix.with_name(
        f"{out_prefix.name}_similarity_heatmap_k{chosen_k}_{chosen_tag}.png"
    ), dpi=150)
    plt.close(fig)

def vectorize_top_keywords(series_of_dicts, top_k=10) -> pd.DataFrame:
    vocab = {}
    for d in series_of_dicts:
        if isinstance(d, dict):
            for k, v in d.items():
                try:
                    vocab[k] = vocab.get(k, 0) + int(v)
                except Exception:
                    pass
    tokens = [t for t, _ in sorted(vocab.items(), key=lambda kv: -kv[1])[:top_k]]
    if not tokens:
        return pd.DataFrame(index=range(len(series_of_dicts)))
    mat = [[float((d or {}).get(tok, 0)) for tok in tokens] for d in series_of_dicts]
    return pd.DataFrame(mat, columns=[f"kw::{t}" for t in tokens])

def coerce_numeric(df: pd.DataFrame, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        else:
            out[c] = np.nan
    return out[cols]

def kmeans_safe_predict(X, k, random_state=42, n_init=20, eps=1e-6):
    # count unique rows at fixed precision
    n_unique = np.unique(np.round(X, 8), axis=0).shape[0]
    X_use = X
    if n_unique < k:
        rng = np.random.RandomState(random_state)
        X_use = X + eps * rng.normal(size=X.shape)
    km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
    return km.fit_predict(X_use)
# ---------- Build dataset ----------
def build_df(profiles_b1, profiles_b2) -> pd.DataFrame:
    df1 = pd.DataFrame(profiles_b1)
    df2 = pd.DataFrame(profiles_b2)
    assert "column_name" in df1.columns and "column_name" in df2.columns, "column_name missing in profiles"
    df = pd.concat([df1, df2], ignore_index=True)
    df["true_label"] = df["column_name"].str.split("::").str[-1]
    return df
# normalization
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MaxAbsScaler

def make_feature_blocks(df: pd.DataFrame, top_k_keywords=10):
    # --- 数值：标准化 ---
    df_num = coerce_numeric(df, NUMERIC_COLS_ALL).fillna(0.0).astype("float64")
    X_num_all = StandardScaler().fit_transform(df_num.values)

    # --- 类别：OHE ---
    df_cat = df[[c for c in CATEG_COLS_ALL if c in df.columns]].fillna("")
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # for older scikit-learn versions
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    X_cat_all = ohe.fit_transform(df_cat) if len(df_cat.columns) else np.zeros((len(df), 0))
    ohe_names_all = list(ohe.get_feature_names_out(df_cat.columns)) if len(df_cat.columns) else []

    # --- 关键词：MaxAbs ---
    df_kw = vectorize_top_keywords(df.get("top_keywords", pd.Series([{}]*len(df))), top_k=top_k_keywords)
    kw_cols_all = list(df_kw.columns)
    X_kw_all = MaxAbsScaler().fit_transform(df_kw.values) if not df_kw.empty else np.zeros((len(df),0))

    # A) 按“分组名”构造
    # def build_by_groups(selected_groups):
    #     use_num, use_cat_bases, use_kw_cols = [], [], []
    #     for g in selected_groups:
    #         if g in ("stats_basic", "distribution", "text_lengths"):
    #             use_num += FEATURE_GROUPS[g]
    #         elif g == "type_info":
    #             use_cat_bases += FEATURE_GROUPS[g]
    #         elif g == "pattern_info":
    #             use_cat_bases += ["dominant_pattern", "first_digit"]
    #             use_num += ["max_digits", "max_decimals"]
    #         elif g == "keywords":
    #             use_kw_cols += kw_cols_all
    #         elif g in ("words_info", "cell_info", "character_info"):
    #             use_num += FEATURE_GROUPS[g]
    #
    #     # 数值切片
    #     num_idx = [i for i,c in enumerate(NUMERIC_COLS_ALL) if c in use_num]
    #     A = X_num_all[:, num_idx] if num_idx else np.zeros((len(df),0))
    #
    #     # 类别切片
    #     if use_cat_bases and ohe_names_all:
    #         mask = np.array([any(n.startswith(base+"_") for base in use_cat_bases) for n in ohe_names_all])
    #         B = X_cat_all[:, mask]
    #     else:
    #         B = np.zeros((len(df),0))
    #
    #     # 关键词切片
    #     if use_kw_cols and kw_cols_all:
    #         kw_idx = [i for i,c in enumerate(kw_cols_all) if c in use_kw_cols]
    #         C = X_kw_all[:, kw_idx] if kw_idx else np.zeros((len(df),0))
    #     else:
    #         C = np.zeros((len(df),0))
    #
    #     return np.hstack([A,B,C])

    # B) 按“具体列名”构造（可用 ALL_COLS 或任意子集）
    def build_by_groups(selected_groups):
        use_num, use_cat_bases, use_kw_cols = [], [], []

        for g in selected_groups:
            if g == "keywords":
                use_kw_cols += kw_cols_all
                continue
            cols = FEATURE_GROUPS.get(g, [])
            for col in cols:
                if col in CATEG_COLS_ALL:
                    use_cat_bases.append(col)
                else:
                    use_num.append(col)

        # numeric slice
        num_idx = [i for i, c in enumerate(NUMERIC_COLS_ALL) if c in use_num]
        A = X_num_all[:, num_idx] if num_idx else np.zeros((len(df), 0))

        # categorical slice (map bases -> OHE columns by prefix)
        if use_cat_bases and ohe_names_all:
            mask = np.array([any(n.startswith(base + "_") for base in use_cat_bases) for n in ohe_names_all])
            B = X_cat_all[:, mask]
        else:
            B = np.zeros((len(df), 0))

        # keyword slice
        if use_kw_cols and kw_cols_all:
            kw_idx = [i for i, c in enumerate(kw_cols_all) if c in use_kw_cols]
            C = X_kw_all[:, kw_idx] if kw_idx else np.zeros((len(df), 0))
        else:
            C = np.zeros((len(df), 0))

        return np.hstack([A, B, C])

    def build_by_columns(selected_columns):
        sel = list(selected_columns)
        # 数值
        num_sel = [c for c in sel if c in NUMERIC_COLS_ALL]
        num_idx = [i for i,c in enumerate(NUMERIC_COLS_ALL) if c in num_sel]
        A = X_num_all[:, num_idx] if num_idx else np.zeros((len(df),0))
        # 类别：把基础列名映射到 OHE 列
        cat_bases = [c for c in sel if c in CATEG_COLS_ALL]
        if cat_bases and ohe_names_all:
            mask = np.array([any(n.startswith(base+"_") for base in cat_bases) for n in ohe_names_all])
            B = X_cat_all[:, mask]
        else:
            B = np.zeros((len(df),0))
        # 关键词：可按列名精确挑选（如 "kw::ipa"）
        kw_sel = [c for c in sel if c in kw_cols_all]
        if kw_sel:
            kw_idx = [i for i,c in enumerate(kw_cols_all) if c in kw_sel]
            C = X_kw_all[:, kw_idx]
        else:
            C = np.zeros((len(df),0))
        return np.hstack([A,B,C])

    return {
        "build_by_groups": build_by_groups,
        "build_by_columns": build_by_columns,
        "ohe_names": ohe_names_all,
        "kw_cols": kw_cols_all,
    }

def build_feature_matrix_and_names(df, selector, *, by="groups", top_k_keywords=10):
    ctx = make_feature_blocks(df, top_k_keywords=top_k_keywords)

    if by == "groups":
        X = ctx["build_by_groups"](selector)
        use_num, use_cat_bases = [], []
        for g in selector:
            cols = FEATURE_GROUPS.get(g, [])
            if g == "keywords":
                continue
            for col in cols:
                if col in CATEG_COLS_ALL:
                    use_cat_bases.append(col)
                else:
                    use_num.append(col)

        num_names = [c for c in NUMERIC_COLS_ALL if c in use_num]
        if use_cat_bases and ctx["ohe_names"]:
            cat_mask = [any(n.startswith(base + "_") for base in use_cat_bases) for n in ctx["ohe_names"]]
            cat_names = [n for n, m in zip(ctx["ohe_names"], cat_mask) if m]
        else:
            cat_names = []
        kw_names = ctx["kw_cols"] if "keywords" in selector else []
        feature_names = num_names + cat_names + kw_names

    elif by == "columns":
        X = ctx["build_by_columns"](selector)
        num_names = [c for c in selector if c in NUMERIC_COLS_ALL]
        cat_bases = [c for c in selector if c in CATEG_COLS_ALL]
        if cat_bases and ctx["ohe_names"]:
            cat_mask = [any(n.startswith(base+"_") for base in cat_bases) for n in ctx["ohe_names"]]
            cat_names = [n for n,m in zip(ctx["ohe_names"], cat_mask) if m]
        else:
            cat_names = []
        kw_names = [c for c in selector if c in ctx["kw_cols"]]
        feature_names = num_names + cat_names + kw_names
    else:
        raise ValueError("by must be 'groups' or 'columns'")

    return X, feature_names

#individual feature test
# ---- Feature set for stats_basic (same list already use)
# STATS_BASIC = NUMERIC_COLS_ALL+CATEG_COLS_ALL

# ---------- Metrics/ utilities----------
def _fit_predict_by_method(X, y_true, method="KMeans", metric="euclidean", k=None):
    if k is None: k = len(np.unique(y_true))

    if method == "KMeans":
        return kmeans_safe_predict(X, k)

    if method == "Agglomerative":
        linkage = "average" if metric == "cosine" else "ward"
        cl = AgglomerativeClustering(n_clusters=k, metric=("cosine" if metric=="cosine" else "euclidean"),
                                     linkage=linkage)
        return cl.fit_predict(X)

    if method == "HDBSCAN":
        if metric == "cosine":
            D = pairwise_distances(X, metric="cosine")
            cl = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=2, metric="precomputed")
            return cl.fit_predict(D)
        else:
            cl = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=2, metric=metric)
            return cl.fit_predict(X)

    raise ValueError(f"Unknown method: {method}")

def pairing_accuracy(true_labels: np.ndarray, pred_clusters: np.ndarray) -> float:
    """
    For each true column name (e.g., 'beer_name'), check if all instances
    (beers_1 and beers_2) fall in the same predicted cluster.
    Returns mean over all true labels.
    """
    acc = []
    tl = pd.Series(true_labels)
    pc = pd.Series(pred_clusters)
    for name, idx in tl.groupby(tl).groups.items():
        clusters = pc.loc[idx].unique()
        acc.append(1.0 if len(clusters) == 1 else 0.0)
    return float(np.mean(acc)) if acc else 0.0


# ---------- Ablation ----------
def run_ablation(df: pd.DataFrame, combos, metric="euclidean", k_grid=None):
    y_true = df["true_label"].values
    true_k = len(np.unique(y_true))
    ctx = make_feature_blocks(df)

    # if no grid provided, fall back to the ground-truth k
    ks = list(k_grid) if k_grid is not None else [true_k]

    rows = []

    for featset in combos:
        # build features by groups
        X = ctx["build_by_groups"](featset)

        # --- KMeans over k-grid ---
        for k in ks:
            y_km = kmeans_safe_predict(X, k)
            rows.append({
                "method": "KMeans",
                "features": "+".join(featset),
                "params": f"k={k}",
                "PairAcc": pairing_accuracy(y_true, y_km),
                "ARI": adjusted_rand_score(y_true, y_km),
                "NMI": normalized_mutual_info_score(y_true, y_km)
            })

        # --- Agglomerative over k-grid ---
        for k in ks:
            try:
                use_metric = "cosine" if metric == "cosine" else "euclidean"
                linkage = "average" if use_metric == "cosine" else "ward"
                agg = AgglomerativeClustering(n_clusters=k, metric=use_metric, linkage=linkage)
                y_agg = agg.fit_predict(X)
                rows.append({
                    "method": "Agglomerative",
                    "features": "+".join(featset),
                    "params": f"k={k},metric={use_metric},linkage={linkage}",
                    "PairAcc": pairing_accuracy(y_true, y_agg),
                    "ARI": adjusted_rand_score(y_true, y_agg),
                    "NMI": normalized_mutual_info_score(y_true, y_agg)
                })
            except Exception:
                pass

        # --- HDBSCAN (no k) ---
        try:
            if metric == "cosine":
                D = pairwise_distances(X, metric="cosine")
                hdb = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=2, metric="precomputed")
                y_hdb = hdb.fit_predict(D)
            else:
                hdb = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=2, metric=metric)
                y_hdb = hdb.fit_predict(X)

            rows.append({
                "method": "HDBSCAN",
                "features": "+".join(featset),
                "params": f"min_cluster_size=2,min_samples=2,metric={metric}",
                "PairAcc": pairing_accuracy(y_true, y_hdb),
                "ARI": adjusted_rand_score(y_true, y_hdb),
                "NMI": normalized_mutual_info_score(y_true, y_hdb)
            })
        except Exception:
            pass

    return pd.DataFrame(rows).sort_values(["ARI","NMI"], ascending=False).reset_index(drop=True)

#--------------heatmap
def _save_heatmap(matrix, row_labels, col_labels, out_path, title="", cmap="coolwarm"):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        M = np.asarray(matrix, dtype=float)
        if M.size == 0:
            print(f"[warn] heatmap skipped: empty matrix for {out_path}")
            return
        M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)

        vmax = np.nanmax(np.abs(M))
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0
        vmin = -vmax

        fig_w = max(6, min(0.18*len(col_labels), 18))
        fig_h = max(6, min(0.25*len(row_labels), 24))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        im = ax.imshow(M, aspect="auto", interpolation="nearest",
                       cmap=cmap, vmin=vmin, vmax=vmax)

        ax.set_title(title)
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_xticklabels(col_labels, rotation=90, fontsize=8)
        ax.set_yticklabels(row_labels, fontsize=8)

        cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cbar.ax.tick_params(labelsize=8)

        plt.tight_layout()
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        print(f"- heatmap saved: {out_path}")
    except Exception as e:
        print(f"[error] failed to save heatmap {out_path}: {e}")

def _save_all_features_heatmap(X, feature_names, row_order, row_labels, cluster_labels, out_path,
                               title="All features heatmap", cmap="viridis"):
    """
    X: (n_samples, n_features)  — 已构建好的特征矩阵
    feature_names: 列名（与 X 列顺序一致）
    row_order: 行排序索引（通常按聚类标签排序：np.argsort(pred)）
    row_labels: 每行对应的显示标签（如 df["column_name"].values）
    cluster_labels: 每行对应的簇 id（pred）
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, Normalize
    from pathlib import Path

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) 排序
    Xv = X[row_order, :]
    row_labels = np.asarray(row_labels)[row_order]
    cl = np.asarray(cluster_labels)[row_order]

    # 2) 列级可视化归一化 [0,1]（仅用于画图，避免 OHE/数值混合时被淹没）
    M = Xv.astype(float).copy()
    for j in range(M.shape[1]):
        col = M[:, j]
        cmin, cmax = np.nanmin(col), np.nanmax(col)
        if not np.isfinite(cmin) or not np.isfinite(cmax) or cmax == cmin:
            M[:, j] = 0.0
        else:
            M[:, j] = (col - cmin) / (cmax - cmin)
    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)

    # 3) 生成簇颜色条（简单循环色）
    uniq = np.unique(cl)
    palette = plt.cm.tab20.colors  # 20 种颜色循环
    color_map = {u: palette[i % len(palette)] for i, u in enumerate(uniq)}
    side_colors = np.array([color_map[u] for u in cl])

    # 4) 画图：左侧窄轴 = 簇颜色条，右侧主体 = 热图
    fig_w = max(10, min(0.16 * len(feature_names) + 1.5, 24))
    fig_h = max(6, min(0.18 * len(row_labels), 24))
    fig = plt.figure(figsize=(fig_w, fig_h))

    # 左侧颜色条轴
    ax_side = plt.axes([0.05, 0.1, 0.02, 0.8])  # [left, bottom, width, height]
    ax_side.imshow(side_colors.reshape(-1, 1, 3), aspect="auto")
    ax_side.set_xticks([])
    ax_side.set_yticks([])
    ax_side.set_title("cluster", fontsize=9)

    # 主热图轴
    ax = plt.axes([0.08, 0.1, 0.86, 0.8])
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(feature_names, rotation=90, fontsize=8)
    ax.set_yticklabels(row_labels, fontsize=8)
    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"- all-features heatmap saved: {out_path}")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity


def plot_column_similarity_heatmap(X, sample_names, out_path=None, metric="cosine"):
    """
    Plots and optionally saves a column-to-column similarity heatmap.

    Parameters
    ----------
    X : array-like, shape (n_columns, n_features)
        Feature matrix describing each column.
    sample_names : list of str
        Names of the columns (same order as rows in X).
    out_path : str or Path, optional
        If given, saves the heatmap image.
    metric : {"cosine", "euclidean"}
        Similarity metric to use.
    """
    # Compute similarity
    if metric == "cosine":
        sim = cosine_similarity(X)
    elif metric == "euclidean":
        from sklearn.metrics import pairwise_distances
        dist = pairwise_distances(X, metric="euclidean")
        # convert distance to similarity (scaled 0–1)
        sim = 1 / (1 + dist)
    else:
        raise ValueError("Unsupported metric")

    sim_df = pd.DataFrame(sim, index=sample_names, columns=sample_names)

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_df, cmap="viridis", xticklabels=True, yticklabels=True, square=True,
                cbar_kws={"label": f"{metric} similarity"})
    plt.title(f"Column-to-column {metric} similarity heatmap", fontsize=14)
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150)
        plt.close()
    else:
        plt.show()


def _guess_group_for_feature_name(name: str) -> str:
    if name.startswith("kw::"):
        return "keywords"
    for base in CATEG_COLS_ALL:
        if name.startswith(base + "_"):
            if base in FEATURE_GROUPS.get("type_info", []):
                return "type_info"
            if base in ["dominant_pattern", "first_digit"]:
                return "pattern_info"
    for g, cols in FEATURE_GROUPS.items():
        if g == "keywords":
            continue
        if isinstance(cols, (list, tuple)) and name in cols:
            return g
    return "other"


def group_variance_report(X, feature_names, out_dir, center_for_cosine=False, topn_feature=20, title_suffix=""):
    """
    计算每个特征组的总体方差占比 + 单特征方差TopN，并保存CSV与条形图。
    X: (n_samples, n_features)  聚类用的特征矩阵
    feature_names: 与X列对齐的特征名
    center_for_cosine: 若聚类/度量更偏cosine，可先对每列做去均值，然后计算方差
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    M = np.asarray(X, dtype=float)

    # 可选：为更贴合cosine，先对每列中心化
    if center_for_cosine:
        M = M - np.nanmean(M, axis=0, keepdims=True)

    # 列方差（以nan安全的方式）
    col_var = np.nanvar(M, axis=0)
    col_var = np.nan_to_num(col_var, nan=0.0, posinf=0.0, neginf=0.0)

    # 归一化看占比
    total = col_var.sum()
    eps = 1e-12
    if total < eps:
        total = eps
    col_share = col_var / total

    # 映射每列 -> 组
    groups = [_guess_group_for_feature_name(fn) for fn in feature_names]

    # 组内聚合
    df_feat = pd.DataFrame({
        "feature": feature_names,
        "group": groups,
        "variance": col_var,
        "share": col_share
    })
    df_group = (df_feat.groupby("group", as_index=False)
                      .agg(group_variance=("variance","sum"),
                           group_share=("share","sum"))
                      .sort_values("group_share", ascending=False))
    df_feat_sorted = df_feat.sort_values("variance", ascending=False)

    # 保存CSV
    p_group = out_dir / f"group_variance{title_suffix}.csv"
    p_feat  = out_dir / f"feature_variance{title_suffix}.csv"
    df_group.to_csv(p_group, index=False)
    df_feat_sorted.to_csv(p_feat, index=False)

    # 画图：组贡献
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(df_group["group"], df_group["group_share"])
    ax.set_xlabel("share of total variance")
    ax.set_ylabel("feature group")
    ax.set_title(f"Group variance share{title_suffix}")
    plt.tight_layout()
    fig.savefig(out_dir / f"group_variance_bar{title_suffix}.png", dpi=150)
    plt.close(fig)

    # 画图：TopN单特征
    top = df_feat_sorted.head(topn_feature).iloc[::-1]  # 倒序画hbar从上到下
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35*len(top))))
    ax.barh(top["feature"], top["variance"])
    ax.set_xlabel("variance")
    ax.set_ylabel("feature")
    ax.set_title(f"Top-{len(top)} feature variance{title_suffix}")
    plt.tight_layout()
    fig.savefig(out_dir / f"feature_variance_top{topn_feature}{title_suffix}.png", dpi=150)
    plt.close(fig)

    print(f"- saved: {p_group}")
    print(f"- saved: {p_feat}")
    print(f"- saved: {out_dir / ('group_variance_bar'+title_suffix+'.png')}")
    print(f"- saved: {out_dir / ('feature_variance_top'+str(topn_feature)+title_suffix+'.png')}")

    return {"by_group": df_group, "by_feature": df_feat_sorted}


def _save_all_features_heatmap(
    X, feature_names, row_order, row_labels, cluster_labels, out_path,
    title="All features heatmap", cmap="viridis", group_labels=None
):
    """
    X: (n_samples, n_features) matrix in the SAME order as feature_names
    feature_names: list[str], len == n_features
    row_order: permutation to sort rows (e.g., np.argsort(pred))
    row_labels: labels for rows BEFORE ordering (we apply row_order inside)
    cluster_labels: 1D array of cluster ids for each row (for left color bar)
    group_labels: optional list[str] for each feature (column) giving its group name.
                  If None, we will guess using _guess_group_for_feature_name().
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- reorder rows by row_order
    Xv = X[row_order, :]
    row_labels = np.asarray(row_labels)[row_order]
    cl = np.asarray(cluster_labels)[row_order]

    # ---- per-column min-max for visibility (only for plotting)
    M = Xv.astype(float).copy()
    for j in range(M.shape[1]):
        col = M[:, j]
        cmin, cmax = np.nanmin(col), np.nanmax(col)
        if not np.isfinite(cmin) or not np.isfinite(cmax) or cmax == cmin:
            M[:, j] = 0.0
        else:
            M[:, j] = (col - cmin) / (cmax - cmin)
    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- left cluster color strip
    uniq = np.unique(cl)
    palette_rows = plt.cm.tab20.colors
    row_color_map = {u: palette_rows[i % len(palette_rows)] for i, u in enumerate(uniq)}
    side_colors = np.array([row_color_map[u] for u in cl])

    # ---- figure layout (extra room on top for group strip)
    fig_w = max(12, min(0.16 * len(feature_names) + 2.0, 26))
    fig_h = max(7, min(0.18 * len(row_labels) + 1.0, 26))
    fig = plt.figure(figsize=(fig_w, fig_h))

    # left strip for clusters
    ax_side = plt.axes([0.05, 0.14, 0.02, 0.76])    # [l, b, w, h]
    ax_side.imshow(side_colors.reshape(-1, 1, 3), aspect="auto")
    ax_side.set_xticks([])
    ax_side.set_yticks([])
    ax_side.set_title("cluster", fontsize=9)

    # main heatmap
    ax = plt.axes([0.08, 0.14, 0.86, 0.76])
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(feature_names, rotation=90, fontsize=7)
    ax.set_yticklabels(row_labels, fontsize=8)
    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.ax.tick_params(labelsize=8)

    # ---- top group color strip + separators
    if group_labels is None:
        group_labels = [_guess_group_for_feature_name(fn) for fn in feature_names]
    group_labels = list(group_labels)

    # top strip axis
    ax_top = plt.axes([0.08, 0.91, 0.86, 0.03])  # very short height
    # build group palette
    groups_order = []
    for g in ["stats_basic", "distribution", "type_info", "pattern_info", "keywords", "other"]:
        if g in group_labels:
            groups_order.append(g)
    # fallback for unseen
    for g in group_labels:
        if g not in groups_order:
            groups_order.append(g)

    palette_cols = plt.cm.Set3.colors  # pleasant group colors
    grp_color_map = {g: palette_cols[i % len(palette_cols)] for i, g in enumerate(groups_order)}
    top_colors = np.array([grp_color_map[g] for g in group_labels]).reshape(1, -1, 3)

    ax_top.imshow(top_colors, aspect="auto")
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    ax_top.set_title("feature group", fontsize=9, pad=2)

    # vertical separators where group changes
    boundaries = []
    for j in range(1, len(group_labels)):
        if group_labels[j] != group_labels[j - 1]:
            boundaries.append(j - 0.5)

    for x in boundaries:
        ax.axvline(x, color="white", lw=1.0, alpha=0.9)
        ax_top.axvline(x, color="white", lw=1.0, alpha=0.9)

    # legend for groups (compact)
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=grp_color_map[g], label=g) for g in groups_order]
    ax_top.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
                  frameon=False, fontsize=8, ncol=1)

    plt.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"- all-features heatmap with groups saved: {out_path}")

# ---------- Micro Ablation ----------
def inspect_stats_basic(
    df: pd.DataFrame,
    out_dir,
    method="KMeans",
    metric="euclidean",
    plot_stats_basic=False,
    draw_all_features_heatmap=True   # 👈 NEW
):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    y_true = df["true_label"].values
    k = len(np.unique(y_true))

    # 只取 stats_basic 里真正存在的列名
    stats_cols = [c for c in FEATURE_GROUPS["stats_basic"] if c in df.columns]
    if not stats_cols:
        print("[inspect_stats_basic] no stats_basic columns found in df; nothing to do.")
        return

    # --- Single: 每次只用一个列名
    rows_single = []
    for f in stats_cols:
        X, _ = build_feature_matrix_and_names(df, [f], by="columns")
        y_pred = _fit_predict_by_method(X, y_true, method=method, metric=metric, k=k)
        rows_single.append({
            "mode": "single", "feature_set": f,
            "PairAcc": pairing_accuracy(y_true, y_pred),
            "ARI": adjusted_rand_score(y_true, y_pred),
            "NMI": normalized_mutual_info_score(y_true, y_pred),
        })
    df_single = pd.DataFrame(rows_single).sort_values(["PairAcc","ARI","NMI"], ascending=False)

    # --- Leave-one-out: ALL − {f}
    rows_loo = []
    for drop_f in stats_cols:
        keep = [x for x in stats_cols if x != drop_f]
        X, _ = build_feature_matrix_and_names(df, keep, by="columns")
        y_pred = _fit_predict_by_method(X, y_true, method=method, metric=metric, k=k)
        rows_loo.append({
            "mode": "leave_one_out", "feature_set": f"all_minus_{drop_f}", "dropped": drop_f,
            "PairAcc": pairing_accuracy(y_true, y_pred),
            "ARI": adjusted_rand_score(y_true, y_pred),
            "NMI": normalized_mutual_info_score(y_true, y_pred),
        })
    df_loo = pd.DataFrame(rows_loo).sort_values(["PairAcc","ARI","NMI"], ascending=False)

    # --- Baseline: ALL stats_basic
    X_all_stats, _ = build_feature_matrix_and_names(df, stats_cols, by="columns")
    y_all = _fit_predict_by_method(X_all_stats, y_true, method=method, metric=metric, k=k)
    base = {
        "PairAcc": pairing_accuracy(y_true, y_all),
        "ARI": adjusted_rand_score(y_true, y_all),
        "NMI": normalized_mutual_info_score(y_true, y_all),
    }

    # --- Delta (importance)
    df_imp = pd.DataFrame([{
        "dropped": r["dropped"],
        "ΔPairAcc": base["PairAcc"] - r["PairAcc"],
        "ΔARI": base["ARI"] - r["ARI"],
        "ΔNMI": base["NMI"] - r["NMI"],
    } for _, r in df_loo.iterrows()]).sort_values(["ΔPairAcc","ΔARI","ΔNMI"], ascending=False)

    # --- Save CSVs
    p_single = out_dir / "stats_basic_single.csv"
    p_loo    = out_dir / "stats_basic_leave_one_out.csv"
    p_imp    = out_dir / "stats_basic_summary.csv"
    df_single.to_csv(p_single, index=False)
    df_loo.to_csv(p_loo, index=False)
    df_imp.to_csv(p_imp, index=False)

    # --- Plots (bars)
    if plot_stats_basic:
        def _plot_single_feature_metric(single_df, out_path, metric_name="ARI", title=None):
            dfp = single_df.sort_values(metric_name, ascending=True)
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.barh(dfp["feature_set"], dfp[metric_name])
            ax.set_xlabel(metric_name); ax.set_ylabel("Feature")
            ax.set_title(title or f"stats_basic single-feature — {metric_name}")
            plt.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)

        def _plot_delta_metric(imp_df, out_path, delta_col="ΔPairAcc", title=None):
            dfp = imp_df.sort_values(delta_col, ascending=True)
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.barh(dfp["dropped"], dfp[delta_col])
            ax.set_xlabel(delta_col); ax.set_ylabel("Dropped feature")
            ax.set_title(title or f"stats_basic importance — {delta_col}")
            plt.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)

        _plot_single_feature_metric(df_single, out_dir / "stats_basic_single_ARI.png",
                                    metric_name="ARI", title="stats_basic single-feature (ARI)")
        _plot_single_feature_metric(df_single, out_dir / "stats_basic_single_NMI.png",
                                    metric_name="NMI", title="stats_basic single-feature (NMI)")
        _plot_delta_metric(df_imp, out_dir / "stats_basic_importance_DeltaPairAcc.png",
                           delta_col="ΔPairAcc", title="stats_basic importance (ΔPairAcc)")
        _plot_delta_metric(df_imp, out_dir / "stats_basic_importance_DeltaARI.png",
                           delta_col="ΔARI", title="stats_basic importance (ΔARI)")

    # --- Heatmap: stats_basic feature matrix (ordered by baseline y_all)
    try:
        order = np.argsort(y_all)
        _save_heatmap(
            X_all_stats[order, :],
            row_labels=df["column_name"].values[order],
            col_labels=stats_cols,
            out_path=out_dir / "stats_basic_feature_matrix_heatmap.png",
            title=f"stats_basic feature matrix — {method}",
            cmap="coolwarm"
        )
    except Exception as e:
        print(f"[warn] failed to save stats_basic feature heatmap: {e}")

    # --- Confusion matrix (baseline ALL stats_basic) + heatmap
    try:
        assign = df[["column_name","true_label"]].copy()
        assign["pred_cluster"] = pd.Series(y_all).astype(str).values
        cm = pd.crosstab(assign["pred_cluster"], assign["true_label"])
        (out_dir / "stats_basic_confusion_matrix.csv").write_text(cm.to_csv())
        _save_heatmap(
            cm.values,
            row_labels=[str(r) for r in cm.index],
            col_labels=[str(c) for c in cm.columns],
            out_path=out_dir / "stats_basic_confusion_matrix_heatmap.png",
            title=f"Confusion Matrix — stats_basic ({method})",
            cmap="Blues"
        )
    except Exception as e:
        print(f"[warn] failed to save stats_basic confusion matrix heatmap: {e}")

    # --- NEW: All-features heatmap (numeric + OHE + keywords) aligned by y_all order
    if draw_all_features_heatmap:
        try:
            # 构建“全列”特征矩阵（按列名方式：ALL_COLS + keywords）
            # 先取数值+类别：ALL_COLS
            X_all_cols, names_all_cols = build_feature_matrix_and_names(df, ALL_COLS, by="columns")
            # 再把 keywords 也拼上（如果你想也看关键词）
            # 这里用 'groups' 方式构造 keywords；如果没有 keywords，返回空块没关系
            X_kw, names_kw = build_feature_matrix_and_names(df, ["keywords"], by="groups")
            # 拼接（列方向）
            if X_kw.shape[1] > 0:
                X_vis = np.hstack([X_all_cols, X_kw])
                names_vis = names_all_cols + names_kw
            else:
                X_vis = X_all_cols
                names_vis = names_all_cols

            order = np.argsort(y_all)  # 用 stats_basic baseline 的顺序对齐
            _save_all_features_heatmap(
                X=X_vis,
                feature_names=names_vis,
                row_order=order,
                row_labels=df["column_name"].values,
                cluster_labels=y_all,   # 用 baseline 的簇 id 上色
                out_path=out_dir / "all_features_heatmap.png",
                title=f"All features heatmap (aligned by stats_basic {method})",
                cmap="viridis"
            )
        except Exception as e:
            print(f"[warn] failed to save all-features heatmap: {e}")

    # --- Console summary
    print("\n[stats_basic] baseline (ALL features):", base)
    print("\n[stats_basic] Single-feature ranking:")
    print(df_single.to_string(index=False))
    print("\n[stats_basic] Leave-one-out (drops when removed) — higher deltas mean more important:")
    print(df_imp.to_string(index=False))
    print("\nSaved files under:", out_dir)

def inspect_best(beers1_path, beers2_path, out_dir, metric="euclidean"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load inputs
    prof1 = load_dictionary_file(Path(beers1_path))
    prof2 = load_dictionary_file(Path(beers2_path))
    df = build_df(extract_rows(prof1), extract_rows(prof2))

    # Run a focused ablation to pick the winner (reuse your combos)
    groups = ["stats_basic","distribution", "words_info", "cell_info","character_info","type_info","pattern_info","keywords"]
    combos = [
        ["stats_basic"],
        ["type_info"],
        ["pattern_info"],
        ["keywords"],
        ["words_info"],
        ["cell_info"],
        ["character_info"],
        ["type_info", "pattern_info"],
        ["stats_basic", "words_info"],
        ["stats_basic", "cell_info"],
        ["stats_basic", "character_info"],
        ["type_info", "pattern_info", "stats_basic"],
        ["type_info", "pattern_info", "stats_basic", "keywords"],
        groups,  # full set
    ]

    res = run_ablation(df, combos, metric=metric)
    res_sorted = res.sort_values(["ARI","NMI"], ascending=False).reset_index(drop=True)
    winner = res_sorted.iloc[0].to_dict()

    # --- Build features for the winner ---
    selected_groups = winner["features"].split("+")
    X, feature_names = build_feature_matrix_and_names(df, selected_groups, by="groups")

    y_true = df["true_label"].values
    ks = range(2, max(3, len(np.unique(y_true)) + 10))  # or any grid you like
    plot_elbow_and_silhouette(
        X,
        ks,
        out_prefix=(out_dir / "inspect_best"),
        sample_names=df["column_name"],  # <<< critical: who each row is
        label_members=True,  # optional: label some names on the bars
        max_labels_per_cluster=40
    )

    # --- Parse k from winner params (Step 4) ---
    method = winner["method"]
    params = winner["params"]

    import re
    def _parse_k(params_str, fallback):
        m = re.search(r"\bk=(\d+)\b", params_str or "")
        return int(m.group(1)) if m else fallback

    # --- Fit winning model with parsed k when applicable ---
    pred = None
    extras = {}

    if method == "KMeans":
        k = _parse_k(params, len(np.unique(y_true)))
        km = KMeans(n_clusters=k, random_state=42, n_init=20).fit(X)
        pred = km.labels_

        # Save centroids with feature names
        centroids = pd.DataFrame(km.cluster_centers_, columns=feature_names)
        centroids.to_csv(out_dir / "kmeans_centroids.csv", index_label="cluster_id")
        extras["kmeans_centroids_path"] = str(out_dir / "kmeans_centroids.csv")

    elif method == "Agglomerative":
        k = _parse_k(params, len(np.unique(y_true)))
        use_metric = "cosine" if "metric=cosine" in params else "euclidean"
        linkage = "average" if "linkage=average" in params else "ward"
        agg = AgglomerativeClustering(
            n_clusters=k,
            metric=use_metric,
            linkage=linkage
        ).fit(X)
        pred = agg.labels_

    elif method == "HDBSCAN":
        use_metric = "cosine" if "metric=cosine" in params else "euclidean"
        if use_metric == "cosine":
            D = pairwise_distances(X, metric="cosine")
            clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=2, metric="precomputed").fit(D)
        else:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=2, metric=use_metric).fit(X)
        pred = clusterer.labels_

        # Diagnostics
        diag = pd.DataFrame({
            "column_name": df["column_name"].values,
            "probability": getattr(clusterer, "probabilities_", np.ones(len(pred))),
            "outlier_score": getattr(clusterer, "outlier_scores_", np.zeros(len(pred))),
            "pred_cluster": pred
        })
        diag.to_csv(out_dir / "hdbscan_scores.csv", index=False)
        extras["hdbscan_scores_path"] = str(out_dir / "hdbscan_scores.csv")

    else:
        raise ValueError(f"Unknown method: {method}")

    # === Variance contribution diagnostics ===
    try:
        group_variance_report(
            X=X,
            feature_names=feature_names,
            out_dir=out_dir,
            center_for_cosine=(metric == "cosine"),
            topn_feature=20,
            title_suffix=f"_{winner['method']}"
        )
    except Exception as e:
        print(f"[warn] group_variance_report failed: {e}")

    # --- Pairing report ---
    pair_rows = []
    for name, sub in df.assign(pred_cluster=pred).groupby("true_label"):
        clusters = sub["pred_cluster"].unique().tolist()
        ok = (len(clusters) == 1)
        pair_rows.append({
            "true_label": name,
            "same_cluster": int(ok),
            "clusters": clusters,
            "members": ", ".join(sub["column_name"].tolist())
        })
    pair_df = pd.DataFrame(pair_rows).sort_values("true_label")
    pair_path = out_dir / "pairing_report.csv"
    pair_df.to_csv(pair_path, index=False)
    print(f"- pairing_report_path: {pair_path}")

    # --- Assignments, breakdown, confusion matrix ---
    assign = df[["column_name","true_label"]].copy()
    assign["pred_cluster"] = pd.Series(pred).astype(str).values
    assign_path = out_dir / "best_assignment.csv"
    assign.to_csv(assign_path, index=False)

    rows = []
    for cid, grp in assign.groupby("pred_cluster"):
        counts = grp["true_label"].value_counts()
        purity = counts.iloc[0] / counts.sum() if len(counts) else 0.0
        rows.append({
            "cluster": cid,
            "size": int(len(grp)),
            "purity": float(purity),
            "labels": json.dumps(counts.to_dict())
        })
    breakdown = pd.DataFrame(rows).sort_values(["purity","size"], ascending=False)
    breakdown_path = out_dir / "cluster_breakdown.csv"
    breakdown.to_csv(breakdown_path, index=False)

    cm = pd.crosstab(assign["pred_cluster"], assign["true_label"])
    cm_path = out_dir / "confusion_matrix.csv"
    cm.to_csv(cm_path)

    # ----- All-features heatmap (numeric + OHE + keywords together) -----
    try:
        order = np.argsort(pred)

        # Group label for each feature column (for the top strip)
        group_labels = [_guess_group_for_feature_name(fn) for fn in feature_names]

        _save_all_features_heatmap(
            X=X,
            feature_names=feature_names,
            row_order=order,
            row_labels=df["column_name"].values,
            cluster_labels=pred,
            out_path=out_dir / "all_features_heatmap.png",
            title=f"All features heatmap (aligned by {winner['method']})",
            cmap="viridis",
            group_labels=group_labels,
        )
    except Exception as e:
        print(f"[warn] failed to save all-features heatmap with groups: {e}")

    try:
        _save_heatmap(
            cm.values,
            row_labels=[str(r) for r in cm.index],
            col_labels=[str(c) for c in cm.columns],
            out_path=out_dir / "confusion_matrix_heatmap.png",
            title="Confusion Matrix (pred_cluster × true_label)",
        )
        print(f"- confusion_matrix_heatmap: {out_dir/'confusion_matrix_heatmap.png'}")
    except Exception as e:
        print(f"[warn] failed to save confusion matrix heatmap: {e}")

    # --- Report ---
    summary = {
        "winner": winner,
        "best_assignment_path": str(assign_path),
        "cluster_breakdown_path": str(breakdown_path),
        "confusion_matrix_path": str(cm_path),
        **extras
    }
    print("\n=== Winner ===")
    print(json.dumps(winner, indent=2))
    print("\nArtifacts:")
    for k,v in summary.items():
        if k.endswith("_path"):
            print(f"- {k}: {v}")
    return summary

# ---------- CLI / path handling ----------
def find_file(in_dir: Path, pattern: str) -> Path:
    matches = sorted(in_dir.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files in {in_dir} matching pattern {pattern!r}")
    if len(matches) > 1:
        print(f"[warn] Multiple files match {pattern!r} in {in_dir}, using: {matches[0]}")
    return matches[0]

from itertools import combinations

# def auto_feature_selection(df, metric="euclidean", max_comb_size=3, topk_single=5):
#     """
#     自动化单特征和组合特征分析，挑选最优组合
#     df: build_df 得到的数据
#     max_comb_size: 最大组合规模（避免指数爆炸）
#     topk_single: 从单特征挑选前k个，作为组合的候选
#     """
#     y_true = df["true_label"].values
#     k = len(np.unique(y_true))
#
#     # --- Step 1: 单特征测试 ---
#     results = []
#     for f in ALL_COLS:
#         if f not in df.columns and not f.startswith("kw::"):
#             continue
#         X, _ = build_feature_matrix_and_names(df, [f], by="columns")
#         y_pred = _fit_predict_by_method(X, y_true, method="KMeans", metric=metric, k=k)
#         results.append({
#             "features": f,
#             "PairAcc": pairing_accuracy(y_true, y_pred),
#             "ARI": adjusted_rand_score(y_true, y_pred),
#             "NMI": normalized_mutual_info_score(y_true, y_pred)
#         })
#     df_single = pd.DataFrame(results).sort_values("ARI", ascending=False)
#     print("\n=== 单特征排名 ===")
#     print(df_single.head(10))
#
#     # --- Step 2: 组合测试 ---
#     # 选出 topk_single 的特征
#     top_feats = df_single.head(topk_single)["features"].tolist()
#
#     combo_results = []
#     for r in range(2, max_comb_size+1):
#         for combo in combinations(top_feats, r):
#             X, _ = build_feature_matrix_and_names(df, list(combo), by="columns")
#             y_pred = _fit_predict_by_method(X, y_true, method="KMeans", metric=metric, k=k)
#             combo_results.append({
#                 "features": "+".join(combo),
#                 "PairAcc": pairing_accuracy(y_true, y_pred),
#                 "ARI": adjusted_rand_score(y_true, y_pred),
#                 "NMI": normalized_mutual_info_score(y_true, y_pred)
#             })
#     df_combo = pd.DataFrame(combo_results).sort_values("ARI", ascending=False)
#     print("\n=== 组合排名 ===")
#     print(df_combo.head(10))
#
#     # --- Step 3: 汇总 ---
#     all_res = pd.concat([df_single, df_combo], ignore_index=True)
#     best = all_res.sort_values("ARI", ascending=False).iloc[0]
#     print("\n=== 最优组合 ===")
#     print(best)
#     return all_res, best
# def evaluate_datasets(dataset_pairs, metric="euclidean", max_comb_size=3, topk_single=5):
#     """
#     dataset_pairs: [(beers1_path, beers2_path, "beers"),
#                     (flights1_path, flights2_path, "flights"),
#                     ...]
#     """
#     all_summaries = []
#
#     for f1, f2, name in dataset_pairs:
#         print(f"\n=== Dataset: {name} ===")
#         prof1 = load_dictionary_file(f1)
#         prof2 = load_dictionary_file(f2)
#         df = build_df(extract_rows(prof1), extract_rows(prof2))
#
#         all_res, best = auto_feature_selection(df, metric=metric,
#                                                max_comb_size=max_comb_size,
#                                                topk_single=topk_single)
#
#         # 保存结果
#         all_res["dataset"] = name
#         all_summaries.append(all_res)
#
#         print(f"\nBest for {name}: {best['features']} "
#               f"(ARI={best['ARI']:.3f}, NMI={best['NMI']:.3f})")
#
#     # 汇总结果
#     summary_df = pd.concat(all_summaries, ignore_index=True)
#     summary_df.to_csv("multi_dataset_feature_selection.csv", index=False)
#     return summary_df
#

def run_group_ablation(df: pd.DataFrame, metric="euclidean", k_grid=None, select_k="grid",
                       save_k_plots=False,
                       plots_dir: str | Path = "./k_plots", label_members_for_plots=False, max_labels_per_cluster=50):

    """
    select_k: "grid" (evaluate all k in k_grid) or "silhouette" (auto-pick best k per combo)
    """
    y_true = df["true_label"].values
    true_k = len(np.unique(y_true))
    ks = list(k_grid) if k_grid is not None else [true_k]

    groups = ["stats_basic","distribution","type_info","pattern_info","keywords",
              "semantic_domain","character_info","words_info","cell_info"
             ]
    all_group_combos = [list(c) for r in range(1, len(groups)+1) for c in combinations(groups, r)]

    rows = []
    ctx = make_feature_blocks(df)

    for featset in all_group_combos:
        X = ctx["build_by_groups"](featset)
        if X.shape[1] == 0:
            continue

        n_unique = np.unique(np.round(X, 8), axis=0).shape[0]

        n_samples = X.shape[0]
        # valid ks: must be <= n_unique
        valid_ks = [k for k in ks if 2 <= k <= min(n_samples, n_unique)]
        if save_k_plots:
            plots_dir = Path(plots_dir)
            plots_dir.mkdir(parents=True, exist_ok=True)

            combo_tag = "+".join(featset).replace("/", "_")
            base_path = plots_dir / combo_tag  # e.g. .../<dataset>/k_plots/stats_basic

            # Let this function do everything:
            # - saves "<base>_elbow_silhouette.png"
            # - saves "<base>_silhouette_plot_k{K}_{tag}.png"
            # - saves "<base>_cluster_members_k{K}_{tag}.csv"
            # - saves "<base>_column_pairs_k{K}_{tag}.csv"
            # - saves "<base>_similarity_heatmap_k{K}_{tag}.png"
            plot_elbow_and_silhouette(
                X,
                valid_ks,
                out_prefix=base_path,  # <-- use base_path (no hard-code!)
                sample_names=df["column_name"],
                k_for_detail="elbow",  # <-- choose k by elbow
                label_members=label_members_for_plots,
                max_labels_per_cluster=max_labels_per_cluster
            )

        if not valid_ks:
            continue
    # draw heatmap between columns with best k in silhoutte
    #     if save_k_plots:
    #         combo_tag = "+".join(featset).replace("/", "_")
    #         base_path = Path(plots_dir) / f"{combo_tag}"
    #
    #         # NEW: similarity heatmap
    #         plot_column_similarity_heatmap(
    #             X,
    #             sample_names=df["column_name"],
    #             out_path=base_path.with_name(base_path.name + "_similarity_heatmap.png"),
    #             metric=metric  # same metric you passed into ablation
    #         )
    #
    #         # existing silhouette/elbow plot
    #         plot_elbow_and_silhouette(
    #             X,
    #             valid_ks,
    #             out_prefix=base_path,
    #             sample_names=df["column_name"],
    #             label_members=False
    #         )

        # ---------- KMEANS ----------
        # if select_k == "silhouette":
        #     best_k, sil = _best_k_by_silhouette(X, valid_ks, metric=metric)
        #     if best_k is not None:
        #         y_km = kmeans_safe_predict(X, best_k)
        #         rows.append({
        #             "method": "KMeans",
        #             "features": "+".join(featset),
        #             "params": f"k={best_k} (silhouette={sil:.4f})",
        #             "PairAcc": pairing_accuracy(y_true, y_km),
        #             "ARI": adjusted_rand_score(y_true, y_km),
        #             "NMI": normalized_mutual_info_score(y_true, y_km)
        #         })
        # else:
        #     for k in valid_ks:
        #         y_km = kmeans_safe_predict(X, k)
        #         rows.append({
        #             "method": "KMeans",
        #             "features": "+".join(featset),
        #             "params": f"k={k}",
        #             "PairAcc": pairing_accuracy(y_true, y_km),
        #             "ARI": adjusted_rand_score(y_true, y_km),
        #             "NMI": normalized_mutual_info_score(y_true, y_km)
        #         }

        # --- KMEANS ---
        if select_k == "silhouette":
            best_k, sil = _best_k_by_silhouette(X, valid_ks, metric=metric)
            if best_k is not None:
                y_km = kmeans_safe_predict(X, best_k)
                rows.append({
                    "method": "KMeans",
                    "features": "+".join(featset),
                    "params": f"k={best_k} (silhouette={sil:.4f})",
                    "PairAcc": pairing_accuracy(y_true, y_km),
                    "ARI": adjusted_rand_score(y_true, y_km),
                    "NMI": normalized_mutual_info_score(y_true, y_km)
                })
        elif select_k == "elbow":
            ks2, inertias = _kmeans_inertia_curve(X, valid_ks)
            if len(ks2):
                elbow_k = _pick_k_by_elbow(ks2, inertias)
                if save_k_plots:
                    combo_tag = "+".join(featset).replace("/", "_")
                    _save_elbow_plot(
                        ks2, inertias,
                        Path(plots_dir) / f"elbow_{combo_tag}.png",
                        title=f"Elbow — {combo_tag}"
                    )
                y_km = kmeans_safe_predict(X, elbow_k)
                rows.append({
                    "method": "KMeans",
                    "features": "+".join(featset),
                    "params": f"k={elbow_k} (elbow)",
                    "PairAcc": pairing_accuracy(y_true, y_km),
                    "ARI": adjusted_rand_score(y_true, y_km),
                    "NMI": normalized_mutual_info_score(y_true, y_km)
                })
        else:
            # original grid sweep
            for k in valid_ks:
                y_km = kmeans_safe_predict(X, k)
                rows.append({
                    "method": "KMeans",
                    "features": "+".join(featset),
                    "params": f"k={k}",
                    "PairAcc": pairing_accuracy(y_true, y_km),
                    "ARI": adjusted_rand_score(y_true, y_km),
                    "NMI": normalized_mutual_info_score(y_true, y_km)
                })

        # ---------- AGGLOMERATIVE ----------
        use_metric = "cosine" if metric == "cosine" else "euclidean"
        linkage = "average" if use_metric == "cosine" else "ward"

        if select_k in ("silhouette", "elbow"):
            # derive best_k the same way you did for KMeans (computed above)
            if select_k == "silhouette":
                best_k, _ = _best_k_by_silhouette(X, valid_ks, metric=metric)
            else:
                ks2, inertias = _kmeans_inertia_curve(X, valid_ks)
                best_k = _pick_k_by_elbow(ks2, inertias) if len(ks2) else None

            if best_k is not None:
                try:
                    agg = AgglomerativeClustering(n_clusters=best_k, metric=use_metric, linkage=linkage)
                    y_agg = agg.fit_predict(X)
                    rows.append({
                        "method": "Agglomerative",
                        "features": "+".join(featset),
                        "params": f"k={best_k},metric={use_metric},linkage={linkage} ({select_k})",
                        "PairAcc": pairing_accuracy(y_true, y_agg),
                        "ARI": adjusted_rand_score(y_true, y_agg),
                        "NMI": normalized_mutual_info_score(y_true, y_agg)
                    })
                except Exception:
                    pass
        else:
            for k in valid_ks:
                try:
                    agg = AgglomerativeClustering(n_clusters=k, metric=use_metric, linkage=linkage)
                    y_agg = agg.fit_predict(X)
                    rows.append({
                        "method": "Agglomerative",
                        "features": "+".join(featset),
                        "params": f"k={k},metric={use_metric},linkage={linkage}",
                        "PairAcc": pairing_accuracy(y_true, y_agg),
                        "ARI": adjusted_rand_score(y_true, y_agg),
                        "NMI": normalized_mutual_info_score(y_true, y_agg)
                    })
                except Exception:
                    pass

        # ---------- HDBSCAN ----------
        # unchanged; it has no k. You can keep it as-is.
        try:
            if metric == "cosine":
                D = pairwise_distances(X, metric="cosine")
                hdb = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=2, metric="precomputed")
                y_hdb = hdb.fit_predict(D)
            else:
                hdb = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=2, metric=metric)
                y_hdb = hdb.fit_predict(X)

            rows.append({
                "method": "HDBSCAN",
                "features": "+".join(featset),
                "params": f"min_cluster_size=2,min_samples=2,metric={metric}",
                "PairAcc": pairing_accuracy(y_true, y_hdb),
                "ARI": adjusted_rand_score(y_true, y_hdb),
                "NMI": normalized_mutual_info_score(y_true, y_hdb)
            })
        except Exception:
            pass

    return pd.DataFrame(rows).sort_values(["ARI","NMI"], ascending=False).reset_index(drop=True)

def auto_feature_selection(
    df,
    metric="euclidean",
    max_comb_size=3,
    topk_single=5,
    out_dir: Path | None = None,
    dataset_name: str | None = None,
):
    """
    Returns (all_results_df, best_row). If out_dir and dataset_name are given,
    saves CSVs under out_dir/dataset_name/.
    """
    y_true = df["true_label"].values
    k = len(np.unique(y_true))

    # --- Step 1: single features ---
    results = []
    for f in ALL_COLS:
        if f not in df.columns and not f.startswith("kw::"):
            continue
        X, _ = build_feature_matrix_and_names(df, [f], by="columns")
        y_pred = _fit_predict_by_method(X, y_true, method="KMeans", metric=metric, k=k)
        results.append({
            "features": f,
            "PairAcc": pairing_accuracy(y_true, y_pred),
            "ARI": adjusted_rand_score(y_true, y_pred),
            "NMI": normalized_mutual_info_score(y_true, y_pred),
            "mode": "single"
        })
    df_single = pd.DataFrame(results).sort_values("ARI", ascending=False)

    # --- Step 2: top-k combos up to max_comb_size ---
    top_feats = df_single.head(topk_single)["features"].tolist()
    combo_rows = []
    for r in range(2, max_comb_size+1):
        for combo in combinations(top_feats, r):
            X, _ = build_feature_matrix_and_names(df, list(combo), by="columns")
            y_pred = _fit_predict_by_method(X, y_true, method="KMeans", metric=metric, k=k)
            combo_rows.append({
                "features": "+".join(combo),
                "PairAcc": pairing_accuracy(y_true, y_pred),
                "ARI": adjusted_rand_score(y_true, y_pred),
                "NMI": normalized_mutual_info_score(y_true, y_pred),
                "mode": f"combo_r{r}"
            })
    df_combo = pd.DataFrame(combo_rows).sort_values("ARI", ascending=False)

    # --- Step 3: merge + pick best ---
    all_res = pd.concat([df_single, df_combo], ignore_index=True)
    best = all_res.sort_values("ARI", ascending=False).iloc[0]

    # --- Save if requested ---
    if out_dir is not None and dataset_name is not None:
        ds_dir = (Path(out_dir) / dataset_name).resolve()
        ds_dir.mkdir(parents=True, exist_ok=True)
        p_single = ds_dir / f"{dataset_name}_single_features.csv"
        p_combo  = ds_dir / f"{dataset_name}_combo_features.csv"
        p_all    = ds_dir / f"{dataset_name}_all_results.csv"
        df_single.to_csv(p_single, index=False)
        df_combo.to_csv(p_combo, index=False)
        all_res.to_csv(p_all, index=False)
        print(f"[ablation] saved single: {p_single}")
        print(f"[ablation] saved combos: {p_combo}")
        print(f"[ablation] saved all:    {p_all}")

    return all_res, best
def evaluate_datasets(
    dataset_pairs,
    metric="euclidean",
    k_min=2,
    k_max=30,
    out_dir: Path | str = "./ablation_out_group",
    label_members_for_plots=False,
    max_labels_per_cluster=50
):
    """
    Runs GROUP-level ablation for each dataset (all non-empty combinations of the 6 groups).
    Saves per-dataset CSVs under out_dir/<dataset>/group_ablation_results.csv
    and a global concatenated summary at out_dir/group_ablation_summary.csv
    """
    out_root = Path(out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[group-ablation] results root: {out_root}")

    k_grid = range(k_min, k_max+1)
    all_results = []

    for f1, f2, name in dataset_pairs:
        print(f"\n=== Dataset: {name} ===")
        prof1 = load_dictionary_file(Path(f1))
        prof2 = load_dictionary_file(Path(f2))
        df = build_df(extract_rows(prof1), extract_rows(prof2))

        # res = run_group_ablation(df, metric=metric, k_grid=k_grid)
        res = run_group_ablation(
            df,
            metric=metric,
            k_grid=range(k_min, k_max + 1),
            select_k="elbow",  # or "silhouette" or "grid"
            save_k_plots=True,
            plots_dir=Path(out_root) / name / "k_plots",
            label_members_for_plots=label_members_for_plots,
            max_labels_per_cluster=max_labels_per_cluster
        )
        res["dataset"] = name
        all_results.append(res)

        ds_dir = out_root / name
        ds_dir.mkdir(parents=True, exist_ok=True)
        ds_csv = ds_dir / "group_ablation_results.csv"
        res.to_csv(ds_csv, index=False)
        print(f"[group-ablation] saved: {ds_csv}")

        # quick peek
        print("Top results:")
        print(res.head(10).to_string(index=False))

    summary_df = pd.concat(all_results, ignore_index=True)
    summary_csv = out_root / "group_ablation_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n[group-ablation] saved summary: {summary_csv}")
    return summary_df

def main():
    # ---- Parse CLI arguments ----
    ap = argparse.ArgumentParser(description="Column profile clustering ablation + inspection")
    # Explicit files
    ap.add_argument("--beers1", type=str, help=".dictionary file for beers_1 column profiles")
    ap.add_argument("--beers2", type=str, help=".dictionary file for beers_2 column profiles")
    # Or folder + patterns
    ap.add_argument("--in_dir", type=str, help="Directory to search for input files")
    ap.add_argument("--pattern1", type=str, default="*beers_1*dictionary")
    ap.add_argument("--pattern2", type=str, default="*beers_2*dictionary")
    # Output
    ap.add_argument("--out_dir", type=str, default=".", help="Directory to save results")
    # Options
    ap.add_argument("--metric", choices=["euclidean","cosine"], default="euclidean")
    ap.add_argument("--topk_keywords", type=int, default=30)
    ap.add_argument("--full", action="store_true", help="Exhaustive feature combos (slow)")
    # Inspections
    ap.add_argument("--inspect", action="store_true", help="Inspect the best clustering in detail")
    ap.add_argument("--inspect_stats_basic", action="store_true", help="Micro-ablation inside stats_basic")
    ap.add_argument("--stats_method", choices=["KMeans","Agglomerative","HDBSCAN"], default="KMeans",
                    help="Clustering method for stats_basic inspection")
    ap.add_argument("--plot_stats_basic", action="store_true",
                    help="Also save a bar chart of ΔPairAcc for stats_basic micro-ablation")
    ap.add_argument("--k_min", type=int, default=2, help="Min k for KMeans/Agglomerative search")
    ap.add_argument("--k_max", type=int, default=30, help="Max k for KMeans/Agglomerative search")

    args = ap.parse_args()

    # ---- Resolve input paths ----
    if args.beers1 and args.beers2:
        f1, f2 = Path(args.beers1).resolve(), Path(args.beers2).resolve()
    elif args.in_dir:
        base = Path(args.in_dir).resolve()
        f1 = find_file(base, args.pattern1)
        f2 = find_file(base, args.pattern2)
    else:
        sys.exit("Provide either --beers1/--beers2 or --in_dir with patterns")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load & build dataframe ----
    prof1, prof2 = load_dictionary_file(f1), load_dictionary_file(f2)
    profiles_b1, profiles_b2 = extract_rows(prof1), extract_rows(prof2)
    df = build_df(profiles_b1, profiles_b2)

    # ---- Optional: Inspect best clustering ----
    if args.inspect:
        inspect_best(f1, f2, out_dir, metric=args.metric)

    # ---- Optional: Micro-ablation inside stats_basic ----
    if args.inspect_stats_basic:
        inspect_stats_basic(
            df,
            out_dir=out_dir,
            method=args.stats_method,
            metric=args.metric,
            plot_stats_basic=args.plot_stats_basic
        )
    # ---- Build combos ----
    groups = ["stats_basic", "distribution", "type_info", "pattern_info", "keywords"]
    if args.full:
        combos = [list(c) for r in range(1, len(groups) + 1) for c in combinations(groups, r)]
    else:
        combos = [
            ["stats_basic"],
            ["type_info"],
            ["pattern_info"],
            ["keywords"],
            ["type_info", "pattern_info"],
            ["type_info", "pattern_info", "stats_basic"],
            ["type_info", "pattern_info", "stats_basic", "keywords"],
            groups,  # full
        ]

    # ---- Run ablation with debug prints ----
    print("\n[debug] entering ablation section...", flush=True)
    print(f"[debug] executing file: {Path(__file__).resolve()}", flush=True)
    print(f"[debug] argv: {sys.argv}", flush=True)
    print(f"[debug] cwd: {Path.cwd()}", flush=True)
    print(f"[debug] out_dir (resolved): {out_dir}", flush=True)
    print(f"[debug] metric: {args.metric}", flush=True)
    print(f"[debug] combos count: {len(combos)}", flush=True)

    k_grid = range(args.k_min, args.k_max + 1)
    print(f"[debug] k_grid: {args.k_min}..{args.k_max}", flush=True)

    res = run_ablation(df, combos, metric=args.metric, k_grid=k_grid)
    print(f"[debug] run_ablation finished. res shape: {res.shape}", flush=True)

    # ---- Save results safely ----
    res_path = (out_dir / "ablation_results.csv").resolve()
    try:
        res_path.parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(res_path, index=False)
        print(f"[ablation] saved: {res_path}", flush=True)
    except Exception as e:
        print(f"[ERROR] could not save results to {res_path}: {e}", flush=True)

    print(res.head(12).to_string(index=False))

    best = res.iloc[0]
    print("\nBest combination:")
    print(f"  Method:   {best['method']}")
    print(f"  Features: {best['features']}")
    print(f"  Params:   {best['params']}")
    print(f"  PairAcc:  {best['PairAcc']:.4f}")
    print(f"  ARI:      {best['ARI']:.4f}")
    print(f"  NMI:      {best['NMI']:.4f}")
    print(f"Saved ablation results to {res_path}")

    # ---- Optional: Micro-ablation inside stats_basic ----
    # if args.inspect_stats_basic:
    #     inspect_stats_basic(df, out_dir, method=args.stats_method, metric=args.metric)
    #
    # # ---- Optional: PairAcc Bar chart ----
    # if args.inspect_stats_basic:
    #     inspect_stats_basic(
    #         df,
    #         out_dir=out_dir,
    #         method=args.stats_method,
    #         metric=args.metric,
    #         plot_stats_basic=args.plot_stats_basic
    #     )


if __name__ == "__main__":
    main()
    # python3 experiment/cluster_feature_ablation_study/cluster_experiment.py \
    #  --in_dir "/Users/veraz/PycharmProjects/DataLakeRuleGeneration/results/Quintet_Match" \
    #  --pattern1 "*beers_1*/column_profile.dictionary" \
    #  --pattern2 "*beers_2*/column_profile.dictionary" \
    #  --out_dir "/Users/veraz/PycharmProjects/DataLakeRuleGeneration/experiment/ablation_results"


# python3 experiment/cluster_feature_ablation_study/cluster_experiment.py \
#     --beers1 "/Users/veraz/PycharmProjects/DataLakeRuleGeneration/results/Quintet_Match/beers_1/column_profile.dictionary" \
#     --beers2 "/Users/veraz/PycharmProjects/DataLakeRuleGeneration/results/Quintet_Match/beers_2/column_profile.dictionary" \
#     --out_dir "/Users/veraz/PycharmProjects/DataLakeRuleGeneration/experiment/ablation_results" \
#     --metric cosine \
#     --inspect
