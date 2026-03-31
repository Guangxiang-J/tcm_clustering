from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture

from .common import (
    DEFAULT_TRAIT_COLUMNS,
    choose_dbscan_eps,
    choose_gmm_k,
    choose_kmeans_k,
    clustering_metrics,
    ensure_dir,
    get_symptoms,
    load_table,
    reduce_x,
    scale_x,
    scatter_2d,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the main dimensionality-reduction and clustering comparison.")
    parser.add_argument("--input", default="data/clean_8_binary.xlsx", help="Path to input symptom file (.xlsx/.xls/.csv).")
    parser.add_argument("--output", default="results/main_analysis", help="Output directory.")
    parser.add_argument("--trait-columns", nargs="+", default=DEFAULT_TRAIT_COLUMNS, help="Trait columns to exclude from symptom clustering.")
    parser.add_argument("--random-state", type=int, default=42, help="Global random seed.")
    parser.add_argument("--kmeans-n-init", type=int, default=50, help="Number of KMeans initializations.")
    parser.add_argument("--gmm-n-init", type=int, default=10, help="Number of GMM initializations.")
    parser.add_argument("--gmm-covariance-type", default="full", help="Covariance type for GaussianMixture.")
    parser.add_argument("--kmeans-k-min", type=int, default=8, help="Minimum K tested for KMeans.")
    parser.add_argument("--kmeans-k-max", type=int, default=30, help="Maximum K tested for KMeans.")
    parser.add_argument("--gmm-k-min", type=int, default=10, help="Minimum K tested for GMM.")
    parser.add_argument("--gmm-k-max", type=int, default=30, help="Maximum K tested for GMM.")
    parser.add_argument("--dbscan-n-neighbors", type=int, default=5, help="k used in the k-distance heuristic for DBSCAN eps.")
    return parser


def run_main_analysis(args: argparse.Namespace) -> Path:
    out_dir = ensure_dir(args.output)
    df = load_table(args.input)
    symptoms = get_symptoms(df, args.trait_columns)
    x = scale_x(symptoms)

    results: list[dict] = []
    best_summary: dict[str, dict] = {}
    reductions = ["PCA", "UMAP", "t-SNE"]
    kmeans_k_range = range(args.kmeans_k_min, args.kmeans_k_max + 1)
    gmm_k_range = range(args.gmm_k_min, args.gmm_k_max + 1)

    for reduction_name in reductions:
        reduced_df, optimal_dim = reduce_x(reduction_name, x, random_state=args.random_state)
        reduced = reduced_df.to_numpy()

        k_kmeans, kmeans_grid = choose_kmeans_k(
            reduced,
            seed=args.random_state,
            k_range=kmeans_k_range,
            n_init=args.kmeans_n_init,
        )
        kmeans_labels = KMeans(
            n_clusters=k_kmeans,
            random_state=args.random_state,
            n_init=args.kmeans_n_init,
        ).fit_predict(reduced)
        results.append(
            {
                "reduction": reduction_name,
                "clustering": "KMeans",
                "optimal_dim": optimal_dim,
                "selected_k": k_kmeans,
                **clustering_metrics(reduced, kmeans_labels),
            }
        )
        kmeans_grid.to_csv(out_dir / f"{reduction_name}_kmeans_k_search.csv", index=False)

        eps = choose_dbscan_eps(reduced, n_neighbors=args.dbscan_n_neighbors)
        dbscan_labels = DBSCAN(eps=eps, min_samples=args.dbscan_n_neighbors).fit_predict(reduced)
        results.append(
            {
                "reduction": reduction_name,
                "clustering": "DBSCAN",
                "optimal_dim": optimal_dim,
                "selected_k": np.nan,
                "eps": eps,
                **clustering_metrics(reduced, dbscan_labels),
            }
        )

        k_gmm, gmm_grid = choose_gmm_k(
            reduced,
            seed=args.random_state,
            k_range=gmm_k_range,
            n_init=args.gmm_n_init,
            covariance_type=args.gmm_covariance_type,
        )
        gmm_labels = GaussianMixture(
            n_components=k_gmm,
            covariance_type=args.gmm_covariance_type,
            random_state=args.random_state,
            n_init=args.gmm_n_init,
        ).fit_predict(reduced)
        results.append(
            {
                "reduction": reduction_name,
                "clustering": "GaussianMixture",
                "optimal_dim": optimal_dim,
                "selected_k": k_gmm,
                **clustering_metrics(reduced, gmm_labels),
            }
        )
        gmm_grid.to_csv(out_dir / f"{reduction_name}_gmm_k_search.csv", index=False)

        best_summary[reduction_name] = {
            "optimal_dim": optimal_dim,
            "kmeans_k": k_kmeans,
            "gmm_k": k_gmm,
            "dbscan_eps": eps,
        }

        scatter_2d(reduced_df.iloc[:, :2], kmeans_labels, f"{reduction_name} + KMeans (k={k_kmeans})", out_dir / f"{reduction_name}_kmeans_scatter.png")
        scatter_2d(reduced_df.iloc[:, :2], gmm_labels, f"{reduction_name} + GMM (k={k_gmm})", out_dir / f"{reduction_name}_gmm_scatter.png")
        scatter_2d(reduced_df.iloc[:, :2], dbscan_labels, f"{reduction_name} + DBSCAN", out_dir / f"{reduction_name}_dbscan_scatter.png")

    pd.DataFrame(results).to_csv(out_dir / "main_analysis_results.csv", index=False)
    write_json(
        out_dir / "run_summary.json",
        {
            "input_file": str(args.input),
            "output_dir": str(args.output),
            "random_state": args.random_state,
            "trait_columns": list(args.trait_columns),
            "kmeans_n_init": args.kmeans_n_init,
            "gmm_n_init": args.gmm_n_init,
            "gmm_covariance_type": args.gmm_covariance_type,
            "best_combinations_from_original_paper": {
                "primary_model": "UMAP + GaussianMixture, K=10",
                "closest_competitor": "UMAP + KMeans, K=15",
            },
            "current_run_summary": best_summary,
        },
    )
    return Path(out_dir)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    output_path = run_main_analysis(args)
    print(f"Main analysis completed: {output_path}")


if __name__ == "__main__":
    main()
