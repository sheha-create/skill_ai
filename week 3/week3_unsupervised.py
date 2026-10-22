"""
Week 3 - Unsupervised Learning + Model Evaluation
Machine Learning & AI Course

Topics: K-Means Clustering, PCA, Elbow Method, Model Saving/Loading

Assignments:
  1. Perform K-Means on Iris dataset and visualize clusters
  2. Apply PCA to reduce dataset dimensions

Mini Project 3: Iris Flower Clustering Project
  - Apply K-Means (k=3)
  - Visualize clusters (scatter plot)
  - Compare predicted clusters vs true labels
"""

import os
import pickle

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

IRIS_PATH = r"C:\Users\Home\Downloads\data3\Iris.csv"
OUTPUT_DIR = r"C:\Users\Home\skillnexis\week 3\output"
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_iris() -> pd.DataFrame:
    section("1. LOAD IRIS DATASET")
    df = pd.read_csv(IRIS_PATH)
    print(f"Shape: {df.shape}")
    df.info()
    print(f"\nSpecies distribution:\n{df['Species'].value_counts()}")
    print(f"\n.describe():\n{df.describe()}")
    return df


def preprocess(df: pd.DataFrame):
    section("2. PREPROCESS + SCALE")
    df = df.drop(columns=["Id"])
    feature_cols = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
    X = df[feature_cols].values
    le = LabelEncoder()
    y_true = le.fit_transform(df["Species"])
    print(f"Encoded species: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"StandardScaler applied. Mean≈0, Std≈1 per feature.")
    return df, feature_cols, X, X_scaled, y_true, le, scaler


def elbow_method(X_scaled) -> int:
    section("3. ELBOW METHOD (choosing K)")
    wcss = []
    k_range = range(1, 11)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        wcss.append(km.inertia_)
        print(f"  k={k:>2}  WCSS={km.inertia_:>10.2f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_range, wcss, "bo-", linewidth=2)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("WCSS (Inertia)")
    ax.set_title("Elbow Method - Iris Dataset")
    ax.set_xticks(list(k_range))
    ax.grid(True, alpha=0.3)
    path = os.path.join(OUTPUT_DIR, "elbow_method.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nSaved -> {path}")
    print("Elbow visible at k=3 (matches the 3 Iris species).")
    return 3


def kmeans_clustering(X_scaled, y_true, feature_cols, le):
    section("4. K-MEANS CLUSTERING (k=3) + VISUALIZATION")
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    y_pred = km.fit_predict(X_scaled)
    print(f"Cluster sizes: {np.bincount(y_pred)}")
    print(f"Cluster centers (scaled):\n{km.cluster_centers_}")

    # 2D scatter: PetalLength vs PetalWidth (most discriminative features)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = ["#e41a1c", "#377eb8", "#4daf4a"]
    for c in range(3):
        mask = y_pred == c
        axes[0].scatter(
            X_scaled[mask, 2], X_scaled[mask, 3],
            c=colors[c], label=f"Cluster {c}", alpha=0.7, edgecolors="k", linewidths=0.3,
        )
    axes[0].scatter(
        km.cluster_centers_[:, 2], km.cluster_centers_[:, 3],
        c="black", marker="X", s=200, edgecolors="yellow", linewidths=1.5,
        label="Centroids",
    )
    axes[0].set_xlabel("PetalLengthCm (scaled)")
    axes[0].set_ylabel("PetalWidthCm (scaled)")
    axes[0].set_title("K-Means Clusters (k=3)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # True labels for comparison
    species_names = le.classes_
    for c, name in enumerate(species_names):
        mask = y_true == c
        axes[1].scatter(
            X_scaled[mask, 2], X_scaled[mask, 3],
            c=colors[c], label=name, alpha=0.7, edgecolors="k", linewidths=0.3,
        )
    axes[1].set_xlabel("PetalLengthCm (scaled)")
    axes[1].set_ylabel("PetalWidthCm (scaled)")
    axes[1].set_title("True Species Labels")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "kmeans_clusters.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved -> {path}")

    # Pairplot of scaled features colored by cluster vs species
    plot_df = pd.DataFrame(X_scaled, columns=feature_cols)
    plot_df["Cluster"] = [f"C{c}" for c in y_pred]
    plot_df["Species"] = le.inverse_transform(y_true)

    pair = sns.pairplot(plot_df, hue="Cluster", palette="Set1", diag_kind="kde", plot_kws={"alpha": 0.6})
    pair.fig.suptitle("Feature Pairs by Predicted Cluster", y=1.02)
    path2 = os.path.join(OUTPUT_DIR, "pairplot_clusters.png")
    pair.savefig(path2, dpi=120)
    plt.close()
    print(f"Saved -> {path2}")

    return km, y_pred


def compare_labels(y_true, y_pred, le) -> None:
    section("5. COMPARE PREDICTED CLUSTERS vs TRUE LABELS")
    species = list(le.classes_)

    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=[f"True {s}" for s in species],
                         columns=[f"Pred C{i}" for i in range(cm.shape[1])])
    print(f"\nConfusion matrix (rows=true, cols=cluster):\n{cm_df}")

    # Map each cluster to the majority true label, then compute accuracy
    mapping = {}
    for c in range(cm.shape[1]):
        mapping[c] = int(np.argmax(cm[:, c]))
    y_mapped = np.array([mapping[c] for c in y_pred])
    acc = accuracy_score(y_true, y_mapped)
    print(f"\nCluster -> species mapping: "
          f"{ {f'C{k}': species[v] for k, v in mapping.items()} }")
    print(f"Accuracy after mapping: {acc:.4f} ({acc*100:.2f}%)")

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"Clusters vs True Labels\n(mapped accuracy = {acc*100:.1f}%)")
    path = os.path.join(OUTPUT_DIR, "cluster_vs_labels.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved -> {path}")


def apply_pca(X_scaled, y_true, le):
    section("6. PCA - DIMENSIONALITY REDUCTION")
    pca_full = PCA()
    pca_full.fit(X_scaled)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    print("\nExplained variance per component:")
    for i, (evr, cv) in enumerate(zip(pca_full.explained_variance_ratio_, cumvar)):
        print(f"  PC{i+1}: {evr:.4f}  (cumulative: {cv:.4f})")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(1, 5), pca_full.explained_variance_ratio_, alpha=0.6, label="Individual")
    ax.step(range(1, 5), cumvar, where="mid", color="red", label="Cumulative")
    ax.axhline(0.95, color="green", linestyle="--", label="95% threshold")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("PCA Explained Variance - Iris")
    ax.set_xticks(range(1, 5))
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = os.path.join(OUTPUT_DIR, "pca_variance.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved -> {path}")

    # Reduce to 2 components and visualize
    pca2 = PCA(n_components=2)
    X_pca = pca2.fit_transform(X_scaled)
    print(f"\nReduced shape: {X_scaled.shape} -> {X_pca.shape}")
    print(f"PC1 + PC2 explain {pca2.explained_variance_ratio_.sum()*100:.2f}% of variance.")

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#e41a1c", "#377eb8", "#4daf4a"]
    for c, name in enumerate(le.classes_):
        mask = y_true == c
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[c], label=name,
                   alpha=0.7, edgecolors="k", linewidths=0.3)
    ax.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title("Iris Dataset - PCA (2D Projection)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path2 = os.path.join(OUTPUT_DIR, "pca_projection.png")
    plt.tight_layout()
    plt.savefig(path2, dpi=150)
    plt.close()
    print(f"Saved -> {path2}")

    # K-Means on PCA features
    km_pca = KMeans(n_clusters=3, random_state=42, n_init=10)
    y_pca_pred = km_pca.fit_predict(X_pca)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for c in range(3):
        mask = y_pca_pred == c
        axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[c],
                        label=f"Cluster {c}", alpha=0.7, edgecolors="k", linewidths=0.3)
    axes[0].scatter(km_pca.cluster_centers_[:, 0], km_pca.cluster_centers_[:, 1],
                    c="black", marker="X", s=200, edgecolors="yellow", linewidths=1.5,
                    label="Centroids")
    axes[0].set_title("K-Means on PCA Features")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for c, name in enumerate(le.classes_):
        mask = y_true == c
        axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[c],
                        label=name, alpha=0.7, edgecolors="k", linewidths=0.3)
    axes[1].set_title("True Labels (PCA space)")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path3 = os.path.join(OUTPUT_DIR, "kmeans_on_pca.png")
    plt.savefig(path3, dpi=150)
    plt.close()
    print(f"Saved -> {path3}")

    return pca2


def save_model(km, scaler, pca2, le) -> None:
    section("7. SAVE MODELS (joblib + pickle)")
    os.makedirs(MODEL_DIR, exist_ok=True)

    artifacts = {"kmeans": km, "scaler": scaler, "pca": pca2, "label_encoder": le}

    joblib_path = os.path.join(MODEL_DIR, "iris_kmeans.joblib")
    joblib.dump(artifacts, joblib_path)
    print(f"Saved (joblib) -> {joblib_path}")

    pkl_path = os.path.join(MODEL_DIR, "iris_kmeans.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(artifacts, f)
    print(f"Saved (pickle) -> {pkl_path}")

    # Reload check
    loaded = joblib.load(joblib_path)
    sample = loaded["scaler"].transform([[5.1, 3.5, 1.4, 0.2]])
    cluster = loaded["kmeans"].predict(sample)[0]
    pca_check = loaded["pca"].transform(sample)
    print(f"\nReload test: [5.1, 3.5, 1.4, 0.2] -> cluster {cluster}, "
          f"PCA shape check {pca_check.shape}")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_iris()
    df, feature_cols, X, X_scaled, y_true, le, scaler = preprocess(df)
    elbow_method(X_scaled)
    km, y_pred = kmeans_clustering(X_scaled, y_true, feature_cols, le)
    compare_labels(y_true, y_pred, le)
    pca2 = apply_pca(X_scaled, y_true, le)
    save_model(km, scaler, pca2, le)


if __name__ == "__main__":
    main()
