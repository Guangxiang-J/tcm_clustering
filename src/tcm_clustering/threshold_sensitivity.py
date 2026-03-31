from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from .common import (
    DEFAULT_TRAIT_COLUMNS,
    build_umap_embedding,
    choose_gmm_k,
    choose_kmeans_k,
    clustering_metrics,
    encode_binary,
    ensure_dir,
    load_table,
    pairwise_scores,
    prevalence_top_symptoms,
    scale_x,
    split_data,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run threshold sensitivity analysis across multiple binarization thresholds.")
    parser.add_argument("--input", default="data/clean_8_COPY.xlsx", help="Path to input symptom file (.xlsx/.xls/.csv).")
    parser.add_argument("--output", default="results/threshold_sensitivity", help="Output directory.")
    parser.add_argument("--trait-columns", nargs="+", default=DEFAULT_TRAIT_COLUMNS, help="Trait columns to exclude from symptom clustering.")
    parser.add_argument("--thresholds", nargs="+", type=int, default=[2, 3, 4], help="Thresholds used to binarize symptom items.")
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
    return parser


def run_one_threshold(symptoms_raw: pd.DataFrame, threshold: int, out_dir: Path, args: argparse.Namespace) -> pd.DataFrame:
    symptoms = encode_binary(symptoms_raw, threshold)
    x_scaled = scale_x(symptoms)

    summary_rows: list[dict] = []
    kmeans_k_range = range(args.kmeans_k_min, args.kmeans_k_max + 1)
    gmm_k_range = range(args.gmm_k_min, args.gmm_k_max + 1)

    for model_name in ["GMM", "KMeans"]:
        best_k_rows: list[dict] = []
        fixed_rows: list[dict] = []
        top_rows: list[pd.DataFrame] = []
        fixed_labels: dict[int, list[int]] = {}

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
                fixed_model = GaussianMixture(
                    n_components=args.gmm_fixed_k,
                    covariance_type=args.gmm_covariance_type,
                    random_state=seed,
                    n_init=args.gmm_n_init,
                )
                labels = fixed_model.fit_predict(embedding)
            else:
                best_k, grid = choose_kmeans_k(
                    embedding,
                    seed=seed,
                    k_range=kmeans_k_range,
                    n_init=args.kmeans_n_init,
                )
                fixed_model = KMeans(n_clusters=args.kmeans_fixed_k, random_state=seed, n_init=args.kmeans_n_init)
                labels = fixed_model.fit_predict(embedding)

            fixed_labels[seed] = labels.tolist()
            best_k_rows.append({"seed": seed, "best_k": best_k})
            fixed_rows.append({"seed": seed, **clustering_metrics(embedding, labels)})

            top_df = prevalence_top_symptoms(symptoms.reset_index(drop=True), labels, args.top_n_symptoms)
            top_df.insert(0, "seed", seed)
            top_rows.append(top_df)
            grid.to_csv(out_dir / f"{model_name.lower()}_seed_{seed:03d}_k_search.csv", index=False)

        pd.DataFrame(best_k_rows).to_csv(out_dir / f"{model_name.lower()}_best_k_by_seed.csv", index=False)
        fixed_df = pd.DataFrame(fixed_rows)
        fixed_df.to_csv(out_dir / f"{model_name.lower()}_fixed_metrics_by_seed.csv", index=False)
        pair_df = pairwise_scores(fixed_labels)
        pair_df.to_csv(out_dir / f"{model_name.lower()}_fixed_pairwise_agreement.csv", index=False)
        pd.concat(top_rows, ignore_index=True).to_csv(out_dir / f"{model_name.lower()}_fixed_top_symptoms.csv", index=False)

        best_mode = int(pd.Series([row["best_k"] for row in best_k_rows]).mode().iloc[0])
        summary_rows.append(
            {
                "threshold": threshold,
                "method": model_name,
                "best_k_mode": best_mode,
                "mean_silhouette": float(fixed_df["silhouette"].mean()),
                "sd_silhouette": float(fixed_df["silhouette"].std()),
                "mean_ch": float(fixed_df["ch"].mean()),
                "mean_db": float(fixed_df["db"].mean()),
                "mean_pairwise_ari": float(pair_df["ari"].mean()),
                "mean_pairwise_nmi": float(pair_df["nmi"].mean()),
            }
        )

    return pd.DataFrame(summary_rows)


def run_threshold_sensitivity(args: argparse.Namespace) -> Path:
    out_dir = ensure_dir(args.output)
    df = load_table(args.input)
    _, symptoms_raw = split_data(df, args.trait_columns)

    all_summaries: list[pd.DataFrame] = []
    for threshold in args.thresholds:
        threshold_dir = ensure_dir(out_dir / f"threshold_{threshold}")
        all_summaries.append(run_one_threshold(symptoms_raw, threshold, threshold_dir, args))

    pd.concat(all_summaries, ignore_index=True).to_csv(out_dir / "threshold_summary.csv", index=False)
    write_json(
        out_dir / "run_summary.json",
        {
            "input_file": str(args.input),
            "output_dir": str(args.output),
            "trait_columns": list(args.trait_columns),
            "thresholds": list(args.thresholds),
            "seeds": list(args.seeds),
            "umap": {
                "n_components": args.umap_n_components,
                "n_neighbors": args.umap_n_neighbors,
                "min_dist": args.umap_min_dist,
            },
            "kmeans_fixed_k": args.kmeans_fixed_k,
            "gmm_fixed_k": args.gmm_fixed_k,
            "kmeans_n_init": args.kmeans_n_init,
            "gmm_n_init": args.gmm_n_init,
            "gmm_covariance_type": args.gmm_covariance_type,
            "top_n_symptoms": args.top_n_symptoms,
        },
    )
    return Path(out_dir)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    output_path = run_threshold_sensitivity(args)
    print(f"Threshold sensitivity analysis completed: {output_path}")


if __name__ == "__main__":
    main()
