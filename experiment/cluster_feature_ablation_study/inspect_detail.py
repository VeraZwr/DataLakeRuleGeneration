from pathlib import Path

from cluster_experiment import inspect_best, run_ablation
from cluster_experiment import load_dictionary_file, extract_rows, build_df, inspect_stats_basic, auto_feature_selection, evaluate_datasets
from itertools import combinations

from pathlib import Path
from cluster_experiment import evaluate_datasets  # adjust import to your filename

dataset_pairs = [
    (Path("/Users/veraz/PycharmProjects/DataLakeRuleGeneration/results/Quintet_Split/beers_1/column_profile.dictionary"),
     Path("/Users/veraz/PycharmProjects/DataLakeRuleGeneration/results/Quintet_Split/beers_2/column_profile.dictionary"),
     "beers"),
    (Path("/Users/veraz/PycharmProjects/DataLakeRuleGeneration/results/Quintet_Split/flights_1/column_profile.dictionary"),
     Path("/Users/veraz/PycharmProjects/DataLakeRuleGeneration/results/Quintet_Split/flights_2/column_profile.dictionary"),
     "flights"),
    (Path("/Users/veraz/PycharmProjects/DataLakeRuleGeneration/results/Quintet_Split/movies_1/column_profile.dictionary"),
     Path("/Users/veraz/PycharmProjects/DataLakeRuleGeneration/results/Quintet_Split/movies_2/column_profile.dictionary"),
     "movies"),
    (Path(
        "/Users/veraz/PycharmProjects/DataLakeRuleGeneration/results/Quintet_Split/hospital_1/column_profile.dictionary"),
     Path(
         "/Users/veraz/PycharmProjects/DataLakeRuleGeneration/results/Quintet_Split/hospital_2/column_profile.dictionary"),
     "hospital"),
    (Path(
        "/Users/veraz/PycharmProjects/DataLakeRuleGeneration/results/Quintet_Split/rayyan_1/column_profile.dictionary"),
     Path(
         "/Users/veraz/PycharmProjects/DataLakeRuleGeneration/results/Quintet_Split/rayyan_2/column_profile.dictionary"),
     "rayyan"),
]
summary = evaluate_datasets(
    dataset_pairs,
    metric="cosine",
    k_min=2,
    k_max=30,
    out_dir="/Users/veraz/PycharmProjects/DataLakeRuleGeneration/experiment/ablation_out_group_9_es_2_figformat_2_bestk_withelbow",
    label_members_for_plots=True,
    max_labels_per_cluster=40
)

print("\n==== Global summary (head) ====")
print(summary.head())

# summary_df = evaluate_datasets(
#     dataset_pairs,
#     metric="cosine",
#     k_min=2,
#     k_max=20,
#     out_dir=Path("/Users/veraz/PycharmProjects/DataLakeRuleGeneration/experiment/ablation_out_group_9")
# )
# print(summary_df.head())

# The function already writes multi_dataset_feature_selection.csv to the CWD.

# --- Paths (set once, reuse) ---
# beers1_path = Path("/Users/veraz/PycharmProjects/DataLakeRuleGeneration/results/Quintet_Match/beers_1/column_profile.dictionary")
# beers2_path = Path("/Users/veraz/PycharmProjects/DataLakeRuleGeneration/results/Quintet_Match/beers_2/column_profile.dictionary")
#
# # --- Load profiles ---
# prof1 = load_dictionary_file(beers1_path)
# prof2 = load_dictionary_file(beers2_path)
#
# # --- Build dataframe ---
# df = build_df(extract_rows(prof1), extract_rows(prof2))
#
# # --- Define feature groups ---
# groups = ["stats_basic","distribution","text_lengths","type_info","pattern_info","keywords"]
#
# # === HERE: generate all possible non-empty combos ===
# combos = [list(c) for r in range(1, len(groups)+1) for c in combinations(groups, r)]
# print(f"Generated {len(combos)} feature group combinations.")
#
# # --- Run ablation over all combos ---
# res = run_ablation(df, combos=combos, metric="cosine")  # or "euclidean"
#
# # --- Inspect results ---
# res_sorted = res.sort_values(["ARI","NMI"], ascending=False)
# print("\nTop results:")
# print(res_sorted.head(10))
#
# # --- Save to CSV for deeper analysis ---
# out_dir = Path("./ablation_out"); out_dir.mkdir(exist_ok=True, parents=True)
# res_sorted.to_csv(out_dir / "all_combos_results.csv", index=False)
# print(f"Saved all results to {out_dir/'all_combos_results.csv'}")
#
# # --- (Optional) Inspect best combo in detail ---
# inspect_best(beers1_path, beers2_path, out_dir, metric="cosine")

#--------------------------------------------------------------

# best overall clustering (cosine distance)
# output best_assignment.csv cluster_breakdown.csv confusion_matrix.csv
# inspect_best(
#     beers1_path="/Users/veraz/PycharmProjects/DataLakeRuleGeneration/results/Quintet_Match/beers_1/column_profile.dictionary",
#     beers2_path="/Users/veraz/PycharmProjects/DataLakeRuleGeneration/results/Quintet_Match/beers_2/column_profile.dictionary",
#     out_dir="/Users/veraz/PycharmProjects/DataLakeRuleGeneration/experiment/ablation_results_detail",
#     metric="cosine"
# )

# ----------best combination
# prof1 = load_dictionary_file("/Users/veraz/PycharmProjects/DataLakeRuleGeneration/results/Quintet_Match/beers_1/column_profile.dictionary")
# prof2 = load_dictionary_file("/Users/veraz/PycharmProjects/DataLakeRuleGeneration/results/Quintet_Match/beers_2/column_profile.dictionary")
# df = build_df(extract_rows(prof1), extract_rows(prof2))
#
# all_res, best = auto_feature_selection(df, metric="euclidean", max_comb_size=10, topk_single=5)

#--------multi datasets
# root_path = "/Users/veraz/PycharmProjects/DataLakeRuleGeneration/results/Quintet_Match/"
# dataset_pairs = [
#     (Path(root_path + "beers_1/column_profile.dictionary"),
#      Path(root_path + "beers_2/column_profile.dictionary"), "beers"),
#     (Path(root_path + "flights_1/column_profile.dictionary"),
#      Path(root_path + "flights_2/column_profile.dictionary"), "flights"),
#     (Path(root_path + "hospital_1/column_profile.dictionary"),
#      Path(root_path + "hospital_2/column_profile.dictionary"), "hospital"),
#     (Path(root_path + "movies_1/column_profile.dictionary"),
#      Path(root_path + "movies_2/column_profile.dictionary"), "movies"),
#     (Path(root_path + "rayyan_1/column_profile.dictionary"),
#      Path(root_path + "rayyan_2/column_profile.dictionary"), "rayyan"),
# ]
#
# summary_df = evaluate_datasets(dataset_pairs, metric="cosine")

# root_path = "/Users/veraz/PycharmProjects/DataLakeRuleGeneration/results/Quintet_Split_test/"
# dataset_pairs = [
#     (Path(root_path + "beers/column_profile.dictionary"),
#      Path(root_path + "beers_2/column_profile.dictionary"), "beers"),
# ]
#
# summary_df = evaluate_datasets(dataset_pairs, metric="cosine")


# -------stats_basic feature importance (KMeans, default metric = euclidean)
# stats_basic_single.csv
# stats_basic_leave_one_out.csv
# stats_basic_summary.csv
# stats_basic_importance.png

# prof1 = load_dictionary_file("/Users/veraz/PycharmProjects/DataLakeRuleGeneration/results/Quintet_Match/beers_1/column_profile.dictionary")
# prof2 = load_dictionary_file("/Users/veraz/PycharmProjects/DataLakeRuleGeneration/results/Quintet_Match/beers_2/column_profile.dictionary")
#
# df = build_df(extract_rows(prof1), extract_rows(prof2))
#
# inspect_stats_basic(
#     df,
#     out_dir="/Users/veraz/PycharmProjects/DataLakeRuleGeneration/experiment/stats_basic_analysis_KMeans",
#     method="KMeans",
#     metric="euclidean",
#     plot_stats_basic=True
# )



# -------Inspect stats_basic feature importance with Agglomerative + cosine
# prof1 = load_dictionary_file("/path/to/beers_1/column_profile.dictionary")
# prof2 = load_dictionary_file("/path/to/beers_2/column_profile.dictionary")
#
# df = build_df(extract_rows(prof1), extract_rows(prof2))
#
# inspect_stats_basic(
#     df,
#     out_dir="./experiment/stats_basic_cosine",
#     method="Agglomerative",
#     metric="cosine",
#     plot_stats_basic=True
# )

#------------Inspect stats_basic feature importance with HDBSCAN + cosine
# prof1 = load_dictionary_file("/Users/veraz/PycharmProjects/DataLakeRuleGeneration/results/Quintet_Match/beers_1/column_profile.dictionary")
# prof2 = load_dictionary_file("/Users/veraz/PycharmProjects/DataLakeRuleGeneration/results/Quintet_Match/beers_2/column_profile.dictionary")
#
# df = build_df(extract_rows(prof1), extract_rows(prof2))
#
# inspect_stats_basic(
#     df,
#     out_dir="/Users/veraz/PycharmProjects/DataLakeRuleGeneration/experiment/stats_basic_analysis_HDBSCAN",
#     method="HDBSCAN",
#     metric="euclidean",
#     plot_stats_basic=True
# )
