from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from .common import (
    DEFAULT_TRAIT_COLUMNS,
    align_labels_to_reference,
    build_umap_embedding,
    choose_gmm_k,
    choose_kmeans_k,
    choose_representative_seed,
    clustering_metrics,
    ensure_dir,
    formula_scores,
    formula_top_symptoms,
    get_symptoms,
    load_table,
    prevalence_top_symptoms,
    scale_x,
    summarize_formula_top,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run repeated-run stability analysis at the primary threshold.")
    parser.add_argument("--input", default="data/clean_8_binary.xlsx", help="Path to input symptom file (.xlsx/.xls/.csv).")
    parser.add_argument("--output", default="results/main_stability_representative", help="Output directory.")
    parser.add_argument("--trait-columns", nargs="+", default=DEFAULT_TRAIT_COLUMNS, help="Trait columns to exclude from symptom clustering.")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(30)), help="Repeated-run seeds.")
    parser.add_argument("--umap-n-components", type=int, default=2, help="UMAP output dimensions.")
    parser.add_argument("--umap-n-neighbors", type=int, default=15, help="UMAP n_neighbors.")
    parser.add_argument("--umap-min-dist", type=float, default=0.1, help="UMAP min_dist.")
    parser.add_argument("--kmeans-fixed-k", type=int, default=15, help="Fixed K used for repeated KMeans runs.")
    parser.add_argument("--gmm-fixed-k", type=int, default=10, help="Fixed K used for repeated GMM runs.")
    parser.add_argument("--kmeans-n-init", type=int, default=50, help="Number of KMeans initializations.")
    parser.add_argument("--gmm-n-init", type=int, default=10, help="Number of GMM initializations.")
    parser.add_argument("--gmm-covariance-type", default="full", help="Covariance type for GaussianMixture.")
    parser.add_argument("--kmeans-k-min", type=int, default=8, help="Minimum K tested for KMeans.")
    parser.add_argument("--kmeans-k-max", type=int, default=30, help="Maximum K tested for KMeans.")
    parser.add_argument("--gmm-k-min", type=int, default=10, help="Minimum K tested for GMM.")
    parser.add_argument("--gmm-k-max", type=int, default=30, help="Maximum K tested for GMM.")
    parser.add_argument("--top-n-symptoms", type=int, default=5, help="Number of top symptoms saved per cluster.")
    parser.add_argument("--low-conf-threshold", type=float, default=0.6, help="Posterior threshold used to flag low-confidence GMM assignments.")
    return parser


def run_model(symptoms: pd.DataFrame, x_scaled: np.ndarray, out_dir: Path, model_name: str, args: argparse.Namespace) -> None:
    best_k_rows: list[dict] = []
    fixed_rows: list[dict] = []
    posterior_rows: list[dict] = []
    fixed_labels: dict[int, list[int]] = {}

    kmeans_k_range = range(args.kmeans_k_min, args.kmeans_k_max + 1)
    gmm_k_range = range(args.gmm_k_min, args.gmm_k_max + 1)

    for seed in args.seeds:
        embedding = build_umap_embedding(
            x_scaled,
            seed=seed,
            n_components=args.umap_n_components,
            n_neighbors=args.umap_n_neighbors,
            min_dist=args.umap_min_dist,
        )

        if model_name == "GMM":
            best_k, grid = choose_gmm_k(
                embedding,
                seed=seed,
                k_range=gmm_k_range,
                n_init=args.gmm_n_init,
                covariance_type=args.gmm_covariance_type,
            )
            grid.to_csv(out_dir / f"seed_{seed:03d}_gmm_k_search.csv", index=False)

            if best_k in grid["k"].values:
                best_score_series = grid.loc[grid["k"] == best_k, "combined_score"]
                best_score = float(best_score_series.iloc[0]) if pd.notna(best_score_series.iloc[0]) else np.nan
            else:
                best_score = np.nan

            best_k_rows.append({"seed": seed, "best_k": best_k, "best_score": best_score})
            model = GaussianMixture(
                n_components=args.gmm_fixed_k,
                covariance_type=args.gmm_covariance_type,
                random_state=seed,
                n_init=args.gmm_n_init,
            )
            labels = model.fit_predict(embedding)
            max_posterior = model.predict_proba(embedding).max(axis=1)
            posterior_rows.append(
                {
                    "seed": seed,
                    "mean_max_posterior": float(np.mean(max_posterior)),
                    "median_max_posterior": float(np.median(max_posterior)),
                    f"low_conf_ratio_{args.low_conf_threshold:.1f}": float(np.mean(max_posterior < args.low_conf_threshold)),
                }
            )
        else:
            best_k, grid = choose_kmeans_k(
                embedding,
                seed=seed,
                k_range=kmeans_k_range,
                n_init=args.kmeans_n_init,
            )
            grid.to_csv(out_dir / f"seed_{seed:03d}_kmeans_k_search.csv", index=False)
            inertia_series = grid.loc[grid["k"] == best_k, "inertia"]
            best_k_rows.append({"seed": seed, "best_k": best_k, "best_score": float(inertia_series.iloc[0])})
            model = KMeans(n_clusters=args.kmeans_fixed_k, random_state=seed, n_init=args.kmeans_n_init)
            labels = model.fit_predict(embedding)

        fixed_labels[seed] = labels.tolist()
        fixed_rows.append({"seed": seed, **clustering_metrics(embedding, labels)})

    best_k_df = pd.DataFrame(best_k_rows)
    fixed_metrics_df = pd.DataFrame(fixed_rows)
    posterior_df = pd.DataFrame(posterior_rows) if posterior_rows else pd.DataFrame()

    best_k_df.to_csv(out_dir / f"{model_name.lower()}_best_k_by_seed.csv", index=False)
    fixed_metrics_df.to_csv(out_dir / f"{model_name.lower()}_fixed_metrics_by_seed.csv", index=False)

    reference_seed, reference_summary_df, pair_df = choose_representative_seed(
        fixed_labels,
        fixed_metrics_df,
        posterior_df=posterior_df if not posterior_df.empty else None,
    )
    pair_df.to_csv(out_dir / f"{model_name.lower()}_fixed_pairwise_agreement.csv", index=False)
    reference_summary_df.to_csv(out_dir / f"{model_name.lower()}_reference_seed_selection_summary.csv", index=False)
    pd.DataFrame([{"model": model_name, "reference_seed": reference_seed}]).to_csv(
        out_dir / f"{model_name.lower()}_selected_reference_seed.csv",
        index=False,
    )

    if not posterior_df.empty:
        posterior_df.to_csv(out_dir / "gmm_fixed_posterior_summary.csv", index=False)

    matched_labels, mapping_df = align_labels_to_reference(fixed_labels, reference_seed)
    mapping_df.to_csv(out_dir / f"{model_name.lower()}_cluster_label_mapping_by_seed.csv", index=False)

    size_rows: list[dict] = []
    prevalence_rows: list[pd.DataFrame] = []
    formula_rows: list[pd.DataFrame] = []
    formula_top_rows: list[pd.DataFrame] = []
    symptoms_reset = symptoms.reset_index(drop=True)

    for seed in args.seeds:
        labels = matched_labels[seed]
        counts = pd.Series(labels).value_counts().sort_index()
        size_rows.extend({"seed": seed, "cluster": int(cluster), "cluster_size": int(size)} for cluster, size in counts.items())

        prevalence_df = prevalence_top_symptoms(symptoms_reset, labels, args.top_n_symptoms)
        prevalence_df.insert(0, "seed", seed)
        prevalence_rows.append(prevalence_df)

        score_df = formula_scores(symptoms_reset, labels)
        score_df.insert(0, "seed", seed)
        formula_rows.append(score_df)

        top_df = formula_top_symptoms(score_df.drop(columns=["seed"]), args.top_n_symptoms)
        top_df.insert(0, "seed", seed)
        formula_top_rows.append(top_df)

    pd.DataFrame(size_rows).to_csv(out_dir / f"{model_name.lower()}_fixed_cluster_sizes.csv", index=False)

    prevalence_all = pd.concat(prevalence_rows, ignore_index=True)
    prevalence_all.to_csv(out_dir / f"{model_name.lower()}_fixed_top_symptoms_prevalence_by_seed.csv", index=False)

    formula_all = pd.concat(formula_rows, ignore_index=True)
    formula_all.to_csv(out_dir / f"{model_name.lower()}_fixed_symptom_formula_scores_by_seed.csv", index=False)

    formula_top_all = pd.concat(formula_top_rows, ignore_index=True)
    formula_top_all.to_csv(out_dir / f"{model_name.lower()}_fixed_top_symptoms_formula_by_seed.csv", index=False)

    formula_summary, final_top = summarize_formula_top(formula_top_all, seed_count=len(args.seeds), top_n=args.top_n_symptoms)
    formula_summary.to_csv(out_dir / f"{model_name.lower()}_fixed_top_symptoms_formula_summary.csv", index=False)
    final_top.to_csv(out_dir / f"{model_name.lower()}_fixed_top_symptoms_formula_final_top{args.top_n_symptoms}.csv", index=False)


def run_main_stability(args: argparse.Namespace) -> Path:
    out_dir = ensure_dir(args.output)
    df = load_table(args.input)
    symptoms = get_symptoms(df, args.trait_columns)
    x_scaled = scale_x(symptoms)

    run_model(symptoms, x_scaled, ensure_dir(out_dir / "gmm"), "GMM", args)
    run_model(symptoms, x_scaled, ensure_dir(out_dir / "kmeans"), "KMeans", args)

    write_json(
        out_dir / "run_summary.json",
        {
            "input_file": str(args.input),
            "output_dir": str(args.output),
            "trait_columns": list(args.trait_columns),
            "seeds": list(args.seeds),
            "umap": {
                "n_components": args.umap_n_components,
                "n_neighbors": args.umap_n_neighbors,
                "min_dist": args.umap_min_dist,
            },
            "reference_rule": "highest mean pairwise ARI; tie-break by mean NMI, posterior (GMM only), silhouette, DB, seed",
            "kmeans_fixed_k": args.kmeans_fixed_k,
            "gmm_fixed_k": args.gmm_fixed_k,
            "kmeans_n_init": args.kmeans_n_init,
            "gmm_n_init": args.gmm_n_init,
            "gmm_covariance_type": args.gmm_covariance_type,
            "top_n_symptoms": args.top_n_symptoms,
            "low_conf_threshold": args.low_conf_threshold,
        },
    )
    return Path(out_dir)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    output_path = run_main_stability(args)
    print(f"Main stability analysis completed: {output_path}")


if __name__ == "__main__":
    main()
