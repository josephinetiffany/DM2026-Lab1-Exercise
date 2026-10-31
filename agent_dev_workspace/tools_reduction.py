import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from langchain_core.tools import tool

def make_tools(session):
    @tool
    def reduce_dimensions_tool(
        method: str = "pca",
        n_components: int = 2,
        perplexity: float = 30.0,
        n_neighbors: int = 15,
        random_state: int = 42
    ) -> dict:
        """Reduces feature matrix dimensions (PCA/t-SNE/UMAP) and plots 2D scatter visualization.

        Generates a 2D scatter plot colored by category label and sets it on session.pending_figure.

        Args:
            method (str): Reduction algorithm ('pca', 'tsne', or 'umap'). Default is 'pca'.
            n_components (int): Number of components/dimensions to reduce to. Default is 2.
            perplexity (float): Perplexity parameter for t-SNE. Default is 30.0.
            n_neighbors (int): Number of neighbors parameter for UMAP. Default is 15.
            random_state (int): Seed for random number generator. Default is 42.

        Returns:
            dict: Summary containing result_id, method, coordinates_shape, explained_variance_ratio (if PCA), and top_coordinates.
        """
        if session.feature_matrix is None:
            return {
                "status": "error",
                "message": "No document-term matrix found. Please run build_dtm_tool first."
            }

        X = session.feature_matrix.toarray() if hasattr(session.feature_matrix, "toarray") else np.asarray(session.feature_matrix)
        method_lower = method.lower()
        explained_variance_ratio = None

        if method_lower == "pca":
            reducer = PCA(n_components=n_components, random_state=random_state)
            coords = reducer.fit_transform(X)
            explained_variance_ratio = [round(float(v), 4) for v in reducer.explained_variance_ratio_]
        elif method_lower == "tsne":
            reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
            coords = reducer.fit_transform(X)
        elif method_lower == "umap":
            reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=random_state)
            coords = reducer.fit_transform(X)
        else:
            return {
                "status": "error",
                "message": f"Unsupported method '{method}'. Choose from 'pca', 'tsne', or 'umap'."
            }

        # Determine category hue labels for scatter plot
        labels = None
        if session.labels is not None:
            labels = session.labels
        elif session.dataframe is not None and "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"].values

        # Mandatory 2D Scatter Plot Creation
        fig, ax = plt.subplots(figsize=(8, 6))
        if labels is not None and len(labels) == len(coords):
            sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=labels, ax=ax, palette="tab10", s=60)
            ax.legend(title="Category", bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax.scatter(coords[:, 0], coords[:, 1], c="skyblue", edgecolors="k", alpha=0.7)

        ax.set_title(f"Dimension Reduction ({method_lower.upper()})")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        fig.tight_layout()

        # Set plot on session.pending_figure per spec requirement
        session.pending_figure = fig

        # Save coordinates in session artifacts
        if session.artifacts is None:
            session.artifacts = {}
        session.artifacts[f"reduced_{method_lower}"] = coords

        result_id = session.next_result_id("reduce_dimensions")

        # Prepare lightweight summary (first 5 coordinates)
        summary_coords = np.round(coords[:5], 4).tolist()

        summary = {
            "result_id": result_id,
            "status": "success",
            "method": method_lower,
            "coordinates_shape": list(coords.shape),
            "top_coordinates": summary_coords
        }

        if explained_variance_ratio is not None:
            summary["explained_variance_ratio"] = explained_variance_ratio

        full_report = {
            "method": method_lower,
            "coordinates": coords.tolist(),
            "explained_variance_ratio": explained_variance_ratio
        }

        session.store_result("reduce_dimensions_tool", {"method": method, "n_components": n_components, "random_state": random_state}, summary, full_report=full_report)
        return summary

    return [reduce_dimensions_tool]
