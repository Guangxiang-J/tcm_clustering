from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from kneed import KneeLocator
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    pairwise_distances,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

DEFAULT_TRAIT_COLUMNS = ["YiDC", "YaDC", "QDC", "PDC", "DHC", "BSC", "SDC", "QSC"]


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file format: {path.suffix}")


def write_json(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def get_symptoms(df: pd.DataFrame, trait_columns: Sequence[str]) -> pd.DataFrame:
    return df.drop(columns=[c for c in trait_columns if c in df.columns], errors="ignore").copy()


def split_data(df: pd.DataFrame, trait_columns: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    traits = df[[c for c in trait_columns if c in df.columns]].copy()
    symptoms = get_symptoms(df, trait_columns)
    return traits, symptoms


def encode_binary(symptoms: pd.DataFrame, threshold: int) -> pd.DataFrame:
    return (symptoms >= threshold).astype(int)


def scale_x(df: pd.DataFrame) -> np.ndarray:
    return RobustScaler().fit_transform(df)


def build_umap_embedding(
    x: np.ndarray,
    seed: int,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> np.ndarray:
    reducer = umap.UMAP(
        n_components=n_components,
        random_state=seed,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
    )
    return reducer.fit_transform(x)


def clustering_metrics(x: np.ndarray, labels: Sequence[int]) -> dict[str, float | int]:
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != -1]

    if len(unique_labels) <= 1:
        return {
            "n_clusters_found": int(len(unique_labels)),
            "silhouette": np.nan,
            "ch": np.nan,
            "db": np.nan,
        }

    mask = labels != -1
    x_use = x[mask] if np.any(~mask) else x
    y_use = labels[mask] if np.any(~mask) else labels

    return {
        "n_clusters_found": int(len(np.unique(y_use))),
        "silhouette": float(silhouette_score(x_use, y_use)),
        "ch": float(calinski_harabasz_score(x_use, y_use)),
        "db": float(davies_bouldin_score(x_use, y_use)),
    }


def choose_kmeans_k(
    x: np.ndarray,
    seed: int,
    k_range: Iterable[int],
    n_init: int,
) -> tuple[int, pd.DataFrame]:
    ks = list(k_range)
    if not ks:
        raise ValueError("k_range for KMeans cannot be empty.")

    inertia_values: list[float] = []
    rows: list[dict] = []
    for k in ks:
        model = KMeans(n_clusters=k, random_state=seed, n_init=n_init)
        labels = model.fit_predict(x)
        inertia_values.append(float(model.inertia_))
        rows.append({"k": k, "inertia": float(model.inertia_), **clustering_metrics(x, labels)})

    knee = KneeLocator(ks, inertia_values, curve="convex", direction="decreasing")
    best_k = int(knee.knee) if knee.knee is not None else ks[int(np.argmin(np.gradient(inertia_values)))]
    return best_k, pd.DataFrame(rows)


def choose_gmm_k(
    x: np.ndarray,
    seed: int,
    k_range: Iterable[int],
    n_init: int,
    covariance_type: str,
) -> tuple[int, pd.DataFrame]:
    ks = list(k_range)
    if not ks:
        raise ValueError("k_range for GMM cannot be empty.")
    if len(ks) < 2:
        raise ValueError("k_range for GMM must contain at least two values.")

    bic_values: list[float] = []
    ll_values: list[float] = []
    for k in ks:
        model = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            random_state=seed,
            n_init=n_init,
        )
        model.fit(x)
        bic_values.append(float(model.bic(x)))
        ll_values.append(float(model.lower_bound_))

    bic_arr = np.asarray(bic_values)
    ll_arr = np.asarray(ll_values)
    ll_increment = np.diff(ll_arr) / np.maximum(np.abs(ll_arr[:-1]), 1e-12)

    bic_norm = (bic_arr[1:] - bic_arr.min()) / max(bic_arr.max() - bic_arr.min(), 1e-12)
    inc_norm = (ll_increment - ll_increment.min()) / max(ll_increment.max() - ll_increment.min(), 1e-12)
    combined_score = 0.5 * bic_norm + 0.5 * (1.0 - inc_norm)

    best_k = int(np.argmin(combined_score) + ks[0])

    rows: list[dict] = []
    for i, k in enumerate(ks):
        rows.append(
            {
                "k": k,
                "bic": float(bic_arr[i]),
                "avg_log_likelihood": float(ll_arr[i]),
                "log_likelihood_increment": np.nan if i == 0 else float(ll_increment[i - 1]),
                "combined_score": np.nan if i == 0 else float(combined_score[i - 1]),
            }
        )
    return best_k, pd.DataFrame(rows)


def choose_dbscan_eps(x: np.ndarray, n_neighbors: int) -> float:
    neighbors = NearestNeighbors(n_neighbors=n_neighbors).fit(x)
    distances, _ = neighbors.kneighbors(x)
    kth_distances = np.sort(distances[:, n_neighbors - 1])
    return float(kth_distances[int(len(kth_distances) * 0.9)])


def optimal_pca_dim(x: np.ndarray, variance_threshold: float = 0.95) -> int:
    pca = PCA().fit(x)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    return int(np.argmax(cumulative >= variance_threshold) + 1)


def optimal_tsne_dim(x: np.ndarray, random_state: int) -> int:
    scores: list[float] = []
    for n_components in (2, 3):
        model = TSNE(
            n_components=n_components,
            perplexity=30,
            learning_rate="auto",
            init="pca",
            random_state=random_state,
        )
        model.fit_transform(x)
        scores.append(float(model.kl_divergence_))
    return int(np.argmin(scores) + 2)


def optimal_umap_dim(x: np.ndarray, random_state: int) -> int:
    original_distances = pairwise_distances(x)
    errors: list[float] = []
    for n_components in range(2, 6):
        embedding = umap.UMAP(
            n_components=n_components,
            random_state=random_state,
            n_neighbors=15,
            min_dist=0.1,
        ).fit_transform(x)
        errors.append(float(np.linalg.norm(original_distances - pairwise_distances(embedding))))
    return int(np.argmin(errors) + 2)


def reduce_x(name: str, x: np.ndarray, random_state: int) -> tuple[pd.DataFrame, int]:
    if name == "PCA":
        n_components = optimal_pca_dim(x)
        embedding = PCA(n_components=n_components, random_state=random_state).fit_transform(x)
    elif name == "UMAP":
        n_components = optimal_umap_dim(x, random_state=random_state)
        embedding = umap.UMAP(
            n_components=n_components,
            random_state=random_state,
            n_neighbors=15,
            min_dist=0.1,
        ).fit_transform(x)
    elif name == "t-SNE":
        n_components = optimal_tsne_dim(x, random_state=random_state)
        embedding = TSNE(
            n_components=n_components,
            perplexity=30,
            learning_rate="auto",
            init="pca",
            random_state=random_state,
        ).fit_transform(x)
    else:
        raise ValueError(f"Unsupported reduction method: {name}")

    columns = [f"{name}_{i + 1}" for i in range(embedding.shape[1])]
    return pd.DataFrame(embedding, columns=columns), n_components


def scatter_2d(df_2d: pd.DataFrame, labels: Sequence[int], title: str, out_file: str | Path) -> None:
    if df_2d.shape[1] < 2:
        return
    plt.figure(figsize=(7, 6))
    plt.scatter(df_2d.iloc[:, 0], df_2d.iloc[:, 1], c=labels, s=12, alpha=0.8, cmap="tab20")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()


def prevalence_top_symptoms(symptoms: pd.DataFrame, labels: Sequence[int], top_n: int) -> pd.DataFrame:
    rows: list[dict] = []
    labels_arr = np.asarray(labels)
    for cluster_id in np.unique(labels_arr):
        prevalence = symptoms.loc[labels_arr == cluster_id].mean(axis=0).sort_values(ascending=False).head(top_n)
        for rank, (symptom, value) in enumerate(prevalence.items(), start=1):
            rows.append(
                {
                    "cluster": int(cluster_id),
                    "rank": rank,
                    "symptom": symptom,
                    "prevalence": float(value),
                }
            )
    return pd.DataFrame(rows)


def formula_scores(symptoms: pd.DataFrame, labels: Sequence[int]) -> pd.DataFrame:
    df = symptoms.copy().reset_index(drop=True)
    df["cluster"] = np.asarray(labels)

    symptom_cols = [c for c in df.columns if c != "cluster"]
    symptom_totals = df[symptom_cols].sum(axis=0).replace(0, np.nan)
    cluster_ids = sorted(df["cluster"].unique())
    n_clusters = len(cluster_ids)

    rows: list[dict] = []
    for cluster_id in cluster_ids:
        cluster_df = df[df["cluster"] == cluster_id]
        cluster_size = len(cluster_df)
        cluster_totals = cluster_df[symptom_cols].sum(axis=0)
        cluster_prevalence = cluster_totals / max(cluster_size, 1)

        contribution = cluster_totals / symptom_totals
        mean_contribution = 1.0 / n_clusters
        specificity = contribution - mean_contribution

        for symptom in symptom_cols:
            rows.append(
                {
                    "cluster": int(cluster_id),
                    "symptom": symptom,
                    "cluster_count": float(cluster_totals.get(symptom, np.nan)),
                    "overall_count": float(symptom_totals.get(symptom, np.nan)),
                    "prevalence": float(cluster_prevalence.get(symptom, np.nan)),
                    "relative_contribution": float(contribution.get(symptom, np.nan)),
                    "cluster_specificity": float(specificity.get(symptom, np.nan)),
                }
            )

    result = pd.DataFrame(rows)
    return result.sort_values(
        by=["cluster", "cluster_specificity", "relative_contribution", "prevalence"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)


def formula_top_symptoms(score_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for cluster_id, sub_df in score_df.groupby("cluster", sort=True):
        top_df = sub_df.sort_values(
            by=["cluster_specificity", "relative_contribution", "prevalence"],
            ascending=[False, False, False],
        ).head(top_n).copy()
        top_df["rank"] = range(1, len(top_df) + 1)
        frames.append(
            top_df[
                [
                    "cluster",
                    "rank",
                    "symptom",
                    "prevalence",
                    "relative_contribution",
                    "cluster_specificity",
                    "cluster_count",
                    "overall_count",
                ]
            ]
        )
    return pd.concat(frames, ignore_index=True)


def pairwise_scores(label_dict: dict[int, Sequence[int]]) -> pd.DataFrame:
    rows: list[dict] = []
    keys = sorted(label_dict)
    for seed_a, seed_b in itertools.combinations(keys, 2):
        rows.append(
            {
                "seed_a": seed_a,
                "seed_b": seed_b,
                "ari": float(adjusted_rand_score(label_dict[seed_a], label_dict[seed_b])),
                "nmi": float(normalized_mutual_info_score(label_dict[seed_a], label_dict[seed_b])),
            }
        )
    return pd.DataFrame(rows)


def summarize_seed_agreement(pair_df: pd.DataFrame) -> pd.DataFrame:
    if pair_df.empty:
        return pd.DataFrame(columns=["seed", "mean_ari", "mean_nmi"])

    left = pair_df[["seed_a", "ari", "nmi"]].rename(columns={"seed_a": "seed"})
    right = pair_df[["seed_b", "ari", "nmi"]].rename(columns={"seed_b": "seed"})
    all_pairs = pd.concat([left, right], ignore_index=True)

    return (
        all_pairs.groupby("seed", as_index=False)
        .agg(mean_ari=("ari", "mean"), mean_nmi=("nmi", "mean"))
        .sort_values(["mean_ari", "mean_nmi", "seed"], ascending=[False, False, True])
        .reset_index(drop=True)
    )


def choose_representative_seed(
    label_dict: dict[int, Sequence[int]],
    fixed_metrics_df: pd.DataFrame,
    posterior_df: pd.DataFrame | None = None,
) -> tuple[int, pd.DataFrame, pd.DataFrame]:
    pair_df = pairwise_scores(label_dict)
    agreement_df = summarize_seed_agreement(pair_df)
    ref_df = fixed_metrics_df.merge(agreement_df, on="seed", how="left")

    if posterior_df is not None and not posterior_df.empty:
        ref_df = ref_df.merge(posterior_df, on="seed", how="left")

    if "mean_max_posterior" in ref_df.columns:
        ref_df = ref_df.sort_values(
            by=["mean_ari", "mean_nmi", "mean_max_posterior", "silhouette", "db", "seed"],
            ascending=[False, False, False, False, True, True],
        )
    else:
        ref_df = ref_df.sort_values(
            by=["mean_ari", "mean_nmi", "silhouette", "db", "seed"],
            ascending=[False, False, False, True, True],
        )

    reference_seed = int(ref_df.iloc[0]["seed"])
    return reference_seed, ref_df.reset_index(drop=True), pair_df


def align_labels_to_reference(
    label_dict: dict[int, Sequence[int]],
    reference_seed: int,
) -> tuple[dict[int, np.ndarray], pd.DataFrame]:
    reference_labels = np.asarray(label_dict[reference_seed])
    reference_clusters = np.sort(np.unique(reference_labels))
    aligned: dict[int, np.ndarray] = {reference_seed: reference_labels.copy()}
    mapping_rows: list[dict] = []

    for seed, labels in label_dict.items():
        labels_arr = np.asarray(labels)
        if seed == reference_seed:
            for cluster_id in reference_clusters:
                mapping_rows.append(
                    {
                        "seed": seed,
                        "original_cluster": int(cluster_id),
                        "matched_cluster": int(cluster_id),
                        "overlap_count": int(np.sum((labels_arr == cluster_id) & (reference_labels == cluster_id))),
                    }
                )
            continue

        current_clusters = np.sort(np.unique(labels_arr))
        contingency = np.zeros((len(current_clusters), len(reference_clusters)), dtype=int)
        for i, current_cluster in enumerate(current_clusters):
            for j, ref_cluster in enumerate(reference_clusters):
                contingency[i, j] = int(np.sum((labels_arr == current_cluster) & (reference_labels == ref_cluster)))

        row_ind, col_ind = linear_sum_assignment(-contingency)
        mapping = {int(current_clusters[i]): int(reference_clusters[j]) for i, j in zip(row_ind, col_ind)}
        remapped = np.array([mapping[int(label)] for label in labels_arr], dtype=int)
        aligned[seed] = remapped

        for i, j in zip(row_ind, col_ind):
            mapping_rows.append(
                {
                    "seed": seed,
                    "original_cluster": int(current_clusters[i]),
                    "matched_cluster": int(reference_clusters[j]),
                    "overlap_count": int(contingency[i, j]),
                }
            )

    return aligned, pd.DataFrame(mapping_rows)


def summarize_formula_top(formula_top_by_seed: pd.DataFrame, seed_count: int, top_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_df = (
        formula_top_by_seed.groupby(["cluster", "symptom"], as_index=False)
        .agg(
            mean_prevalence=("prevalence", "mean"),
            sd_prevalence=("prevalence", "std"),
            mean_relative_contribution=("relative_contribution", "mean"),
            sd_relative_contribution=("relative_contribution", "std"),
            mean_cluster_specificity=("cluster_specificity", "mean"),
            sd_cluster_specificity=("cluster_specificity", "std"),
            selection_frequency=("seed", "count"),
            mean_rank=("rank", "mean"),
        )
    )
    summary_df["selection_frequency"] = summary_df["selection_frequency"] / seed_count

    final_frames: list[pd.DataFrame] = []
    for cluster_id, sub_df in summary_df.groupby("cluster", sort=True):
        ranked = sub_df.sort_values(
            by=["mean_cluster_specificity", "mean_relative_contribution", "selection_frequency", "mean_prevalence"],
            ascending=[False, False, False, False],
        ).head(top_n).copy()
        ranked["final_rank"] = range(1, len(ranked) + 1)
        final_frames.append(ranked)

    final_df = pd.concat(final_frames, ignore_index=True)
    return summary_df, final_df
